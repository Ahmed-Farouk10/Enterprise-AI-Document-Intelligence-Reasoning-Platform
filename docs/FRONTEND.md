# Frontend Documentation

## Overview

The DocuCentric frontend is a modern Next.js 16 application with a clean, professional UI built using shadcn/ui components and Tailwind CSS.

---

## Quick Start

### Development Mode

```bash
cd frontend

# Install dependencies
npm install

# Copy environment file
cp .env.example .env.local

# Start development server
npm run dev
```

Access at: http://localhost:3000

### Docker Mode

```bash
# From project root
docker-compose up frontend

# Or with full stack
docker-compose up -d
```

Access at: http://localhost:3000

---

## Environment Configuration

### Required Variables (.env.local)

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NODE_ENV=development
```

### Available Scripts

```bash
npm run dev          # Start development server
npm run build        # Build for production
npm run start        # Start production server
npm run lint         # Run ESLint
```

---

## Architecture

### Directory Structure

```
frontend/
├── src/
│   ├── app/                      # Next.js App Router
│   │   ├── (app)/               # Authenticated routes
│   │   │   ├── dashboard/       # Chat interface
│   │   │   └── knowledge-graph/ # Graph visualization
│   │   ├── layout.tsx           # Root layout
│   │   └── page.tsx             # Landing page
│   ├── components/              # React components
│   │   ├── ui/                  # shadcn/ui primitives
│   │   └── app-sidebar.tsx      # Main navigation
│   ├── hooks/                   # Custom React hooks
│   │   ├── use-chat.ts          # Chat state management
│   │   └── use-documents.ts     # Document management
│   ├── lib/                     # Utilities
│   │   ├── api-client.ts        # Axios wrapper
│   │   └── api/                 # API endpoint definitions
│   └── types/                   # TypeScript interfaces
├── public/                      # Static assets
├── .env.local                   # Environment variables
└── package.json
```

---

## API Integration

### Configuration

The frontend communicates with the backend via:

```typescript
// src/lib/api-client.ts
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
```

### Endpoints Used

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/chat/sessions` | GET, POST | List/create chat sessions |
| `/api/chat/sessions/:id` | GET, DELETE | Get/delete session |
| `/api/chat/sessions/:id/stream` | POST | SSE streaming chat |
| `/api/documents` | GET | List documents |
| `/api/documents/upload` | POST | Upload document |
| `/api/documents/:id` | GET, DELETE | Get/delete document |
| `/api/documents/:id/status` | GET | Check processing status |
| `/api/documents/:id/download` | GET | Download document |

---

## Key Components

### 1. Landing Page (`src/app/page.tsx`)

- Hero section with animated background
- "Launch Platform" button navigates to dashboard
- Professional branding

### 2. Dashboard (`src/app/(app)/dashboard/page.tsx`)

- Chat session management
- Real-time message streaming (SSE)
- Document selection and context
- Message history and persistence

### 3. Sidebar (`src/components/app-sidebar.tsx`)

- Navigation between dashboard and knowledge graph
- Document list with status indicators
- Chat history
- User menu

### 4. Chat Interface

- Message input with validation
- Streaming response display
- Reasoning steps visualization
- Verification warnings
- Loading states and skeletons

---

## State Management

### React Query (TanStack Query)

Used for:
- Data fetching and caching
- Automatic refetching
- Optimistic updates
- Loading/error states

```typescript
// Example: useChat hook
const { sessions, createSession, sendMessage } = useChat()
```

### Local State

- Component state for UI interactions
- Form validation with Zod
- Toast notifications with Sonner

---

## Styling

### Tailwind CSS v4

- Utility-first CSS framework
- Dark mode by default
- Custom color palette in `globals.css`
- Responsive design with breakpoints

### shadcn/ui Components

Professional, accessible UI components:

```typescript
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
```

---

## Development Tips

### 1. Hot Reload

Changes to files in `src/` automatically reload:

```bash
npm run dev  # Auto-reload enabled
```

### 2. API Proxy (Optional)

If CORS issues, add to `next.config.ts`:

```typescript
const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://localhost:8000/api/:path*',
      },
    ]
  },
}
```

### 3. Debugging

Enable verbose logging:

```typescript
// In browser console
localStorage.setItem('debug', '*')
```

### 4. React DevTools

Install [React Developer Tools](https://chrome.google.com/webstore/detail/react-developer-tools) for component inspection.

---

## Testing

### Run Tests

```bash
npm test
```

### Run Linter

```bash
npm run lint
```

---

## Common Issues

### Issue: Can't Connect to Backend

**Solution:**
1. Check `NEXT_PUBLIC_API_URL` in `.env.local`
2. Verify backend is running: `curl http://localhost:8000/health`
3. Restart frontend: `npm run dev`

### Issue: CORS Errors

**Solution:**
1. Backend CORS is configured with `allow_origins=["*"]` for development
2. If issues persist, add rewrite to `next.config.ts` (see above)

### Issue: Build Fails

**Solution:**
```bash
# Clear cache
rm -rf .next node_modules

# Reinstall
npm install

# Rebuild
npm run build
```

---

## Deployment

### Production Build

```bash
npm run build
npm run start
```

### Docker Production

```bash
docker-compose --profile production up -d
```

---

## Contributing

1. Follow TypeScript strict mode
2. Use ESLint rules
3. Add tests for new components
4. Update documentation

---

**For more details, see:**
- [Next.js Documentation](https://nextjs.org/docs)
- [shadcn/ui Documentation](https://ui.shadcn.com)
- [Tailwind CSS Documentation](https://tailwindcss.com)
