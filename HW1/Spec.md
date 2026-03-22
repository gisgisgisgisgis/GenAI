# Objective
Build a custom GPT website using the OpenAI API. The website should let users create a personalized language model experience tailored to specific needs.
Begin with a concise checklist (3-7 bullets) of what you will do; keep items conceptual, not implementation-level.
# Requirements
## Core Features
- Allow users to select the GPT model to use, such as GPT-3 or GPT-4.
- Allow customization of system prompts to guide model behavior.
- Allow customization of commonly used parameters, including temperature, max tokens, and similar settings.
- Support both streaming and non-streaming responses.
- Include short-term memory so conversations can retain context.
# Security Notes
- Do not expose the OpenAI API key in frontend code.
- Use environment variables and server-side code to keep API credentials secure.
- When deploying the website, protect environment variables and API keys using secure storage and access methods.
- Never print, embed, or request real secrets in examples; use placeholders for API keys, tokens, and environment values.
- For any action that could expose secrets or sensitive configuration, stop and propose a safer alternative.
# Output Expectations
- Provide guidance and implementation details for creating the website described above.
- Keep all sensitive credentials protected throughout development and deployment.
- If providing code, state assumptions, follow common project style, and include minimal validation steps to verify key flows such as authentication, streaming, and memory behavior.
- Structure the response with clear sections for architecture, backend, frontend, memory handling, streaming support, deployment, and security.
- Keep the response practical and concise, expanding only where needed for implementation clarity.