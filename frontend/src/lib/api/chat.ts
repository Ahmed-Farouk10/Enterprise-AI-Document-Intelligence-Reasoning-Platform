import { ApiClient } from '../api-client'
import { ChatMessage, ChatSession } from '@/types'

/**
 * Chat API endpoints
 */
export const ChatAPI = {
    /**
     * Get all chat sessions
     */
    async getSessions() {
        return ApiClient.get<ChatSession[]>('/api/chat/sessions')
    },

    /**
     * Get a single chat session with messages
     */
    async getSession(sessionId: string) {
        return ApiClient.get<ChatSession>(`/api/chat/sessions/${sessionId}`)
    },

    /**
     * Create a new chat session
     */
    async createSession(title?: string, documentIds?: string[]) {
        return ApiClient.post<ChatSession>('/api/chat/sessions', {
            title,
            document_ids: documentIds,
        })
    },

    /**
     * Send a message in a chat session
     */
    async sendMessage(sessionId: string, content: string, documentIds?: string[]) {
        return ApiClient.post<ChatMessage>(`/api/chat/sessions/${sessionId}/messages`, {
            content,
            document_ids: documentIds,
        })
    },

    /**
     * Delete a chat session
     */
    async deleteSession(sessionId: string) {
        return ApiClient.delete<void>(`/api/chat/sessions/${sessionId}`)
    },

    /**
     * Stream chat response (for real-time responses)
     */
    async streamMessage(
        sessionId: string,
        content: string,
        onChunk: (chunk: any) => void
    ): Promise<void> {
        const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

        try {
            const response = await fetch(`${API_URL}/api/chat/sessions/${sessionId}/stream`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ content })
            });

            if (!response.ok) throw new Error(response.statusText);

            const reader = response.body?.getReader();
            const decoder = new TextDecoder();

            if (!reader) return;

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value);
                // Handle multiple SSE messages in one chunk
                const lines = chunk.split('\n\n');

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6));
                            onChunk(data);
                        } catch (e) {
                            console.error('Error parsing SSE chunk', e);
                        }
                    }
                }
            }
        } catch (error) {
            console.error('Stream error', error);
            throw error;
        }
    },
}
