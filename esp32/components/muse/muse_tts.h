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

#pragma once

/*
 * Spoken replies through Gemini TTS: one reply's text in, mono PCM at
 * muse_tts_rate() out, streamed as Gemini synthesizes it. Runs its own task; the chat session
 * starts a reply and pulls the audio. Off unless CONFIG_MUSE_GEMINI_TTS_KEY
 * is set.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    MUSE_TTS_IDLE,
    MUSE_TTS_RUNNING,   /* audio is still coming, or still buffered */
    MUSE_TTS_DONE,      /* all of it has been read */
    MUSE_TTS_FAILED,    /* stopped early, and what came has been read */
} muse_tts_state_t;

/* Starts speaking `text`, dropping any reply still going. False if TTS is off
 * (no key) or couldn't start: show the reply as text instead. */
bool muse_tts_start(const char *text);

/* The reply's sample rate (Gemini's own, 24 kHz unless it says otherwise). */
int muse_tts_rate(void);

/* Up to `max` samples of the reply, without blocking. */
size_t muse_tts_read(int16_t *out, size_t max);

muse_tts_state_t muse_tts_state(void);

/* Stops the reply in progress, if any. */
void muse_tts_cancel(void);

#ifdef __cplusplus
}
#endif
