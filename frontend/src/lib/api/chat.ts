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
        onChunk: (chunk: string) => void,
        documentIds?: string[]
    ) {
        // This would use EventSource or WebSocket for streaming
        // Placeholder implementation
        return ApiClient.post<ChatMessage>(`/api/chat/sessions/${sessionId}/stream`, {
            content,
            document_ids: documentIds,
        })
    },
}
