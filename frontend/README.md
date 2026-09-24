# QueryMind Frontend

Next.js 16 + Clerk frontend for QueryMind.

## Required environment

Create `.env.local` in this folder:

```env
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=your_clerk_publishable_key
CLERK_SECRET_KEY=your_clerk_secret_key
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
```

Never commit `.env.local`.

## Run

```powershell
npm install
npm run dev
```

Open http://localhost:3000.

## Authentication

- Email/password and Google are handled by Clerk.
- New users are sent to `/face-register` after Clerk sign-up.
- Face ID is a custom YuNet + SFace biometric flow, not a Clerk passkey.
- Existing users can choose **Sign in with Face ID** from `/sign-in`.

## Chat flow

The browser calls same-origin Next.js API routes. Those routes require a signed-in Clerk session and then call FastAPI:

`Clerk -> Next.js API -> FastAPI -> Supabase/Chroma/Groq`

This prevents the browser from supplying an arbitrary `user_email`.

## Supported uploads

PDF, DOCX, PPTX, XLSX, CSV, TXT, PNG, JPG/JPEG, WEBP, BMP.

The backend enforces a 20 MB file limit.

## Face ID note

The custom face matcher uses OpenCV YuNet for detection and SFace for embeddings. It is a similarity matcher and does not provide hardware-backed WebAuthn/passkey security or strong liveness detection. For a production security-sensitive product, keep email/Google authentication as the stronger fallback and add a liveness/anti-spoofing layer.
