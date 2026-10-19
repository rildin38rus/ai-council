# Security policy

AI Council v0.2 is a local prototype. It has **no authentication**. Start it only on loopback (`127.0.0.1`) and **never expose it to the public Internet** without adding access control, TLS, rate limiting, and a security review.

Keep API credentials exclusively in a local `.env` file. `.env` and SQLite state are excluded by `.gitignore`. Do not open issues containing credentials, medical information, or personal data. Rotate any accidentally exposed keys immediately.

OpenAI/Gemini API calls send the prompt and selected room context to external providers. Confirm the relevant privacy and data processing requirements before sending sensitive material. The demo mode does not call external models.

To report a potential vulnerability, please contact the project maintainer privately rather than posting exploitation details publicly.
