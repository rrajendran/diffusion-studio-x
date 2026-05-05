Conversational Image Generation with Ollama LLM
Currently, the app directly sends the user's input to the chosen image generation provider (e.g., Hugging Face, Ollama CLI). This makes it difficult to say "make it blue" since the underlying image model doesn't understand conversational context.

To mimic the ChatGPT + DALL-E experience, we will introduce a Conversational LLM layer using Ollama.

Proposed Changes
1. Add Conversational LLM Configuration
Add a new setting in the Sidebar for configuring the "Understanding Model" (the Text LLM).

Default: llama3 (or any configurable Ollama model).
This will be separate from the "Image Model" setting.
2. Conversational Intermediary in useChat.js
When a user sends a message, rather than directly calling generateImage, we will:

Construct a prompt including the system instructions, the chat history, and the user's latest query.
The System Prompt will instruct the LLM:
"You are an AI assistant helping a user generate images. Base changes on the previous image prompt. Construct a full, detailed image generation prompt. Respond with JSON: { 'response': '...', 'imagePrompt': '...' }"

Send this payload to the Ollama /api/chat endpoint.
Extract the conversational response and the modified imagePrompt.
Call generateImage using the imagePrompt (if provided).
Display the LLM's conversational text alongside the generated image.
3. Image Editing Workflow
Through the LLM, "editing" works by rewriting the entire prompt with the new modifiers (e.g., "A red car" -> "A blue car"). This is exactly how DALL-E 3 handles dialogue-based editing natively.

Implementation Details
[MODIFY] src/components/Sidebar.jsx
Add an input/select for llmModel configuration.
[MODIFY] src/store/chatStore.js
Update loadConfig, saveConfig, and newChatObj to support llmModel.
[MODIFY] src/hooks/useChat.js
Insert the Ollama LLM interpretation step before calling generateImage.
Handle JSON parsing of the LLM output.
Support appending assistant messages with both text content and the resulting imageUrl.
Open Questions
IMPORTANT

Are we assuming Ollama is running locally on port 11434 for the conversational text model? I will make a direct HTTP call to http://localhost:11434/api/chat from the browser. Let me know if that works for you.

Also, does "prompt-rewriting via LLM context" sufficiently cover your requirement for "capability to edit the image"? (e.g., understanding "make it blue" based on the previous message). Or do you specifically need pixel-based image-to-image pipeline support?

Verification Plan
Send queries like "A futuristic city in the rain".
Follow up with "Make it day time" and ensure the new image is generated accurately by the LLM modifying the generation prompt rather than failing.
