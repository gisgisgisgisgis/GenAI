<<<<<<< HEAD
# GenAI
=======
# Custom Groq Website

## Checklist
- Define a secure client-server architecture with server-side Groq calls.
- Implement configurable model, system prompt, and generation parameters.
- Support both streaming and non-streaming chat responses.
- Add short-term memory so each session retains conversation context.
- Validate core flows (health, chat, streaming, and memory reset).

## Assumptions
- You will run this project in Node.js 18+.
- A Groq API key is available as an environment variable.
- Browser clients talk only to this backend and never to Groq directly.

## Architecture
- Frontend: static HTML/CSS/JS served from `public/`.
- Backend: Express API in `server.js`.
- Groq integration: server-side only via `groq-sdk`.
- Session memory: in-memory `Map` keyed by a browser session ID stored in localStorage.

## Backend
- `POST /api/chat`
  - Accepts model, system prompt, user message, and parameters.
  - Supports `stream: true` (SSE-style chunks) and `stream: false` (single JSON reply).
  - Stores short-term memory (last 20 turns total messages).
- `POST /api/memory/reset`
  - Clears short-term memory for the active session.
- `GET /api/health`
  - Simple health endpoint.

## Frontend
- Model selector includes Groq-supported model options.
- System prompt text area allows behavior customization.
- Parameter controls:
  - temperature
  - max tokens
  - top p
  - presence penalty
  - frequency penalty
- Streaming mode toggle switches between live token rendering and one-shot response.
- Reset button clears server-side conversation memory for current session.

## Memory Handling
- Client keeps a persistent `sessionId` in localStorage.
- Server stores message history by `sessionId`.
- Memory is short-term and process-local:
  - Survives multiple requests while server runs.
  - Clears if the server restarts.

## Streaming Support
- Server returns `text/event-stream` when streaming is enabled.
- Frontend reads chunks from `ReadableStream` and appends assistant text incrementally.
- Non-streaming mode returns JSON with full reply text.

## Security
- API key is read only from environment (`GROQ_API_KEY`).
- No API keys are embedded in frontend code.
- `.env` is ignored by git; `.env.example` provides placeholders only.
- Input is minimally validated (session ID sanitization, required message checks).

## Deployment
- Set environment variables securely in your platform (not in source files).
- Recommended hosting patterns:
  - Containerized Node service
  - Managed platform with secret manager (for example, cloud app service + secret store)
- Ensure HTTPS in production.

## Local Run
1. Install dependencies:
   ```bash
   npm install
   ```
2. Create env file:
   ```bash
   cp .env.example .env
   ```
3. Edit `.env` and set `GROQ_API_KEY` with your real key.
4. Start server:
   ```bash
   npm start
   ```
5. Open `http://localhost:3000`.

## Minimal Validation Steps
1. Health check:
   - Open `http://localhost:3000/api/health`
   - Expect `{ "ok": true }`.
2. Non-streaming:
   - Uncheck Streaming mode.
   - Send a message and confirm one complete assistant bubble appears.
3. Streaming:
   - Enable Streaming mode.
   - Send a message and confirm text appears progressively.
4. Memory behavior:
   - Ask follow-up questions referencing prior context.
   - Confirm model remembers recent turns.
   - Click reset memory and verify context is no longer retained.
>>>>>>> Initial commit
