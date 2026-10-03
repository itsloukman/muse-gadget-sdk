/*
 * Copyright (c) 2026 Loukman.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Gemini TTS over its Interactions API (ai.google.dev/gemini-api/docs/
 * speech-generation): POST /v1beta/interactions with "stream": true answers
 * with server-sent events, each audio "step.delta" carrying base64 16-bit PCM
 * and its "sample_rate". It sends 24 kHz even when asked for 16, so the rate
 * is passed on (muse_tts_rate) and the session resamples as it plays.
 */
#include "muse_tts.h"

#include <string.h>

#include "cJSON.h"
#include "esp_crt_bundle.h"
#include "esp_heap_caps.h"
#include "esp_http_client.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/idf_additions.h"
#include "freertos/stream_buffer.h"
#include "freertos/task.h"
#include "mbedtls/base64.h"
#include "sdkconfig.h"

static const char *TAG = "muse_tts";

#define TTS_URL "https://generativelanguage.googleapis.com/v1beta/interactions"
#define TTS_RATE 24000              /* what Gemini sends; each chunk says so */
#define PCM_BUF (64 * 1024)         /* 2 s of 16 kHz audio between Gemini and the speaker */
#define LINE_MAX_BYTES (256 * 1024) /* one SSE line: an audio chunk in base64 */
#define TEXT_MAX 1024
#define READ_CHUNK 2048
#define TIMEOUT_MS 15000

static StreamBufferHandle_t s_pcm;
static TaskHandle_t s_task;
static char *s_text;
static char *s_line;
static uint8_t *s_raw;
static volatile muse_tts_state_t s_state = MUSE_TTS_IDLE;
static volatile uint32_t s_gen;   /* bumped to cancel the request in flight */
static volatile bool s_busy;      /* the task is between notification and done */
static volatile int s_rate = TTS_RATE;

static void *big(size_t n)
{
    return heap_caps_malloc(n, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
}

/* The first {"type": "audio", "data": "<base64>", ...} anywhere in an event. */
static const cJSON *find_audio(const cJSON *j)
{
    if (!cJSON_IsObject(j) && !cJSON_IsArray(j)) {
        return NULL;
    }
    if (cJSON_IsObject(j)) {
        const char *type = cJSON_GetStringValue(cJSON_GetObjectItem(j, "type"));
        if (type && !strcmp(type, "audio") && cJSON_GetStringValue(cJSON_GetObjectItem(j, "data"))) {
            return j;
        }
    }
    for (const cJSON *c = j->child; c; c = c->next) {
        const cJSON *a = find_audio(c);
        if (a) {
            return a;
        }
    }
    return NULL;
}

/* Queues PCM for the speaker, waiting while the buffer is full. False if cancelled. */
static bool push(const uint8_t *p, size_t n, uint32_t gen)
{
    while (n) {
        if (gen != s_gen) {
            return false;
        }
        size_t sent = xStreamBufferSend(s_pcm, p, n, pdMS_TO_TICKS(100));
        p += sent;
        n -= sent;
    }
    return true;
}

/* One SSE line. False to stop: cancelled, or Gemini reported an error. */
static bool on_line(char *line, uint32_t gen, uint8_t *carry, bool *has_carry, size_t *bytes)
{
    if (strncmp(line, "data:", 5)) {
        return true;   /* "event:", comments, blank lines */
    }
    cJSON *j = cJSON_Parse(line + 5);
    if (!j) {
        return true;
    }
    const cJSON *err = cJSON_GetObjectItem(j, "error");
    if (err) {
        char *s = cJSON_PrintUnformatted(err);
        ESP_LOGW(TAG, "gemini error: %.200s", s ? s : "?");
        cJSON_free(s);
        cJSON_Delete(j);
        return false;
    }
    const cJSON *audio = find_audio(j);
    const char *b64 = audio ? cJSON_GetStringValue(cJSON_GetObjectItem(audio, "data")) : NULL;
    const cJSON *rate = audio ? cJSON_GetObjectItem(audio, "sample_rate") : NULL;
    if (cJSON_IsNumber(rate) && rate->valueint >= 8000 && rate->valueint <= 48000) {
        s_rate = rate->valueint;
    }
    bool ok = true;
    if (b64) {
        size_t n = 0;
        uint8_t *out = s_raw + 1;   /* room for a byte carried from the last chunk */
        if (mbedtls_base64_decode(out, LINE_MAX_BYTES / 4 * 3, &n, (const uint8_t *)b64, strlen(b64)) == 0 && n) {
            if (*has_carry) {
                *--out = *carry;
                n++;
            }
            *has_carry = n & 1;
            if (*has_carry) {
                *carry = out[--n];   /* keep samples whole */
            }
            *bytes += n;
            ok = push(out, n, gen);
        }
    }
    cJSON_Delete(j);
    return ok;
}

static char *request_body(const char *text)
{
    cJSON *root = cJSON_CreateObject();
    cJSON_AddStringToObject(root, "model", CONFIG_MUSE_GEMINI_TTS_MODEL);
    cJSON *content = cJSON_CreateObject();
    cJSON_AddStringToObject(content, "type", "text");
    cJSON_AddStringToObject(content, "text", text);
    cJSON *contents = cJSON_CreateArray();
    cJSON_AddItemToArray(contents, content);
    cJSON *turn = cJSON_CreateObject();
    cJSON_AddStringToObject(turn, "type", "user_input");
    cJSON_AddItemToObject(turn, "content", contents);
    cJSON *input = cJSON_AddArrayToObject(root, "input");
    cJSON_AddItemToArray(input, turn);
    cJSON *fmt = cJSON_AddObjectToObject(root, "response_format");
    cJSON_AddStringToObject(fmt, "type", "audio");
    cJSON_AddStringToObject(fmt, "mime_type", "audio/l16");
    cJSON_AddNumberToObject(fmt, "sample_rate", TTS_RATE);
    cJSON *voice = cJSON_CreateObject();
    cJSON_AddStringToObject(voice, "voice", CONFIG_MUSE_GEMINI_TTS_VOICE);
    cJSON *speech = cJSON_AddArrayToObject(cJSON_AddObjectToObject(root, "generation_config"), "speech_config");
    cJSON_AddItemToArray(speech, voice);
    cJSON_AddTrueToObject(root, "stream");
    char *body = cJSON_PrintUnformatted(root);
    cJSON_Delete(root);
    return body;
}

static bool speak(uint32_t gen)
{
    char *body = request_body(s_text);
    if (!body) {
        return false;
    }
    esp_http_client_config_t cfg = {
        .url = TTS_URL,
        .method = HTTP_METHOD_POST,
        .timeout_ms = TIMEOUT_MS,
        .buffer_size = 4096,
        .buffer_size_tx = 2048,
        .crt_bundle_attach = esp_crt_bundle_attach,
    };
    esp_http_client_handle_t c = esp_http_client_init(&cfg);
    if (!c) {
        cJSON_free(body);
        return false;
    }
    esp_http_client_set_header(c, "x-goog-api-key", CONFIG_MUSE_GEMINI_TTS_KEY);
    esp_http_client_set_header(c, "Content-Type", "application/json");
    esp_http_client_set_header(c, "Accept", "text/event-stream");
    int64_t t0 = esp_timer_get_time();
    size_t len = strlen(body);
    bool ok = esp_http_client_open(c, (int)len) == ESP_OK && esp_http_client_write(c, body, (int)len) == (int)len;
    cJSON_free(body);
    int status = 0;
    if (ok) {
        esp_http_client_fetch_headers(c);
        status = esp_http_client_get_status_code(c);
        ok = status == 200;
    }
    if (!ok) {
        char err[300] = "";
        if (status) {
            int n = esp_http_client_read(c, err, sizeof(err) - 1);
            err[n > 0 ? n : 0] = 0;
        }
        ESP_LOGW(TAG, "request failed (HTTP %d) %s", status, err);
        esp_http_client_cleanup(c);
        return false;
    }

    char buf[READ_CHUNK];
    size_t line_len = 0, bytes = 0;
    uint8_t carry = 0;
    bool has_carry = false, first = true;
    for (;;) {
        int n = esp_http_client_read(c, buf, sizeof(buf));
        if (n <= 0 || gen != s_gen) {
            ok = n == 0 && gen == s_gen;
            break;
        }
        for (int i = 0; i < n && ok; i++) {
            if (buf[i] == '\n') {
                s_line[line_len] = 0;
                if (line_len && s_line[line_len - 1] == '\r') {
                    s_line[line_len - 1] = 0;
                }
                size_t before = bytes;
                ok = on_line(s_line, gen, &carry, &has_carry, &bytes);
                if (first && bytes > before) {
                    first = false;
                    ESP_LOGI(TAG, "first audio after %lld ms", (long long)(esp_timer_get_time() - t0) / 1000);
                }
                line_len = 0;
            } else if (line_len < LINE_MAX_BYTES - 1) {
                s_line[line_len++] = buf[i];
            }
        }
        if (!ok) {
            break;
        }
    }
    esp_http_client_cleanup(c);
    ESP_LOGI(TAG, "%s: %.2fs of speech in %lld ms", ok ? "done" : "stopped", (double)bytes / 2 / TTS_RATE,
             (long long)(esp_timer_get_time() - t0) / 1000);
    return ok && bytes;
}

static void tts_task(void *arg)
{
    (void)arg;
    for (;;) {
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        uint32_t gen = s_gen;
        bool ok = speak(gen);
        if (gen == s_gen) {
            s_state = ok ? MUSE_TTS_DONE : MUSE_TTS_FAILED;
        }
        s_busy = false;
    }
}

static bool init_once(void)
{
    if (s_task) {
        return true;
    }
    s_pcm = xStreamBufferCreateWithCaps(PCM_BUF, 1, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    s_text = big(TEXT_MAX);
    s_line = big(LINE_MAX_BYTES);
    s_raw = big(LINE_MAX_BYTES / 4 * 3 + 2);
    if (!s_pcm || !s_text || !s_line || !s_raw) {
        ESP_LOGE(TAG, "no memory");
        return false;
    }
    /* TLS and JSON: the stack can live in PSRAM (CONFIG_SPIRAM_ALLOW_STACK_EXTERNAL_MEMORY). */
    if (xTaskCreatePinnedToCoreWithCaps(tts_task, "muse_tts", 8192, NULL, 4, &s_task, tskNO_AFFINITY,
                                        MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT) != pdPASS) {
        s_task = NULL;
        ESP_LOGE(TAG, "no task");
        return false;
    }
    return true;
}

void muse_tts_cancel(void)
{
    s_gen++;
    if (s_state == MUSE_TTS_RUNNING) {
        s_state = MUSE_TTS_IDLE;
    }
}

bool muse_tts_start(const char *text)
{
    if (!CONFIG_MUSE_GEMINI_TTS_KEY[0] || !text || !text[0] || !init_once()) {
        return false;
    }
    muse_tts_cancel();
    /* A cancelled request lets go of the buffer within one 100 ms send. */
    for (int i = 0; s_busy && i < 50; i++) {
        vTaskDelay(pdMS_TO_TICKS(20));
    }
    if (s_busy) {
        ESP_LOGW(TAG, "previous reply still stopping");
        return false;
    }
    xStreamBufferReset(s_pcm);
    s_rate = TTS_RATE;
    strlcpy(s_text, text, TEXT_MAX);
    s_state = MUSE_TTS_RUNNING;
    s_busy = true;
    xTaskNotifyGive(s_task);
    return true;
}

int muse_tts_rate(void)
{
    return s_rate;
}

size_t muse_tts_read(int16_t *out, size_t max)
{
    if (!s_pcm) {
        return 0;
    }
    return xStreamBufferReceive(s_pcm, out, max * sizeof(int16_t), 0) / sizeof(int16_t);
}

muse_tts_state_t muse_tts_state(void)
{
    /* Done or failed, it isn't over until the speaker has taken the rest. */
    if ((s_state == MUSE_TTS_DONE || s_state == MUSE_TTS_FAILED) && s_pcm && xStreamBufferBytesAvailable(s_pcm)) {
        return MUSE_TTS_RUNNING;
    }
    return s_state;
}
