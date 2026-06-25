# Chat Session Updates

## 1. Overview
This document consolidates the prior chat-history and file-tracking notes into one place and documents the additive metrics now stored in the chat session JSON data.

The implementation keeps the current chat workflow, file-processing flow, retrieval pipeline, and memory logic intact. Existing session files remain readable, and the new data is stored as optional metadata so older records continue to work.

## 2. Chat Session Tracking
- Persistent chat history remains stored under `chat_history/chat_sessions.json`.
- Existing visible chats keep the same structure and behavior.
- Each visible chat session now supports extra optional metadata fields for richer history tracking.
- Session updates remain thread-safe and additive.
- The UI sidebar shows the saved chat history section so users can switch between sessions without losing context.
- The chat history section includes the active thread label and keeps the selected conversation in sync with the current session.

## 3. File Tracking
- Chunk-tracking continues to be written to `logs/chunked_files.json`.
- Existing chunk-tracking output is unchanged.
- Embedding audit data is now written to `logs/chunked_files.json` so document processing history can be correlated with chat activity without touching chat history storage.

## 4. Response Time Logging
For every assistant turn, the session store can now capture:
- Query timestamp
- Response start timestamp
- Response end timestamp
- Total response latency in milliseconds
- Model name used
- Session ID
- Query ID

The data is stored alongside the existing message history, with the combined turn data recorded in a backward-compatible way.
- In the UI, the assistant answer is generated with a visible `Thinking & Retrieving...` state so users can see when the response pipeline is actively working.
- The latency caption shown after generation continues to report the search query and selected model.

## 5. Embedding Performance Logging
For every processed file, the session store now captures:
- File name
- File path
- File size
- Chunk count
- Embedding model used
- Embedding start timestamp
- Embedding end timestamp
- Total embedding duration in milliseconds
- Status (`success` or `failed`)

This data is written to the `embedding_files` section inside `logs/chunked_files.json` so it stays colocated with file tracking without appearing in chat history.

## 6. Session JSON Structure
A visible chat session can now include optional fields such as:
```json
{
  "id": "uuid",
  "title": "New Chat",
  "created_at": "2026-06-24T10:00:00+00:00",
  "updated_at": "2026-06-24T10:00:10+00:00",
  "messages": [
    {
      "role": "user",
      "content": "Hello",
      "query_id": "uuid",
      "query_timestamp": "2026-06-24T10:00:00.123+00:00",
      "session_id": "uuid"
    },
    {
      "role": "assistant",
      "content": "Hi there",
      "query_id": "uuid",
      "session_id": "uuid",
      "response_metrics": {
        "response_start_time": "2026-06-24T10:00:01.000+00:00",
        "response_end_time": "2026-06-24T10:00:01.900+00:00",
        "response_latency_ms": 900,
        "model_name": "qwen3.5:9b"
      }
    }
  ],
  "turns": [
    {
      "query_id": "uuid",
      "session_id": "uuid",
      "user_query": "Hello",
      "query_timestamp": "2026-06-24T10:00:00.123+00:00",
      "response": "Hi there",
      "response_metrics": {
        "response_start_time": "2026-06-24T10:00:01.000+00:00",
        "response_end_time": "2026-06-24T10:00:01.900+00:00",
        "response_latency_ms": 900,
        "model_name": "qwen3.5:9b"
      }
    }
  ]
}
```

The embedding audit entries are stored separately in `logs/chunked_files.json` under `embedding_files`.

## 7. System Prompt Updates
The system prompt guidance remains centralized in `llm/prompt.py`.

The existing prompt hardening notes from the earlier documentation are still relevant:
- The main RAG prompt prioritizes retrieved context as the primary source of truth.
- The assistant is instructed not to invent unsupported facts.
- Missing information, conflicting sources, and follow-up questions are handled explicitly.
- The rewrite prompt remains unchanged because it only reformulates chat history into a standalone question.

## 8. Migration Notes
- Existing `chat_history/chat_sessions.json` files remain readable.
- Sessions missing the new fields are normalized on load.
- The file-tracking log is updated automatically when embedding metrics are first written.
- No existing endpoints or chat flows need to change.
- The UI-specific changes were applied in `interfaces/webui.py`, which mirrors the same chat-history and generation-status behavior as the main app flow.

## 9. Backward Compatibility Notes
- Existing APIs remain unchanged.
- Existing response formats remain unchanged.
- Existing chat workflows remain unchanged.
- Existing file-processing logic remains unchanged.
- Existing retrieval and memory behavior remain unchanged.
- Missing new metadata fields are handled gracefully.
- Old session files continue to load and save without requiring a migration step.

## 10. Fix the Chat history ui 
- In old ui after click on "New chat" you have to select menually new chat from dropsown
- After update, "New chat" will bring you to new chat 
- Now you can delete and rename then chat