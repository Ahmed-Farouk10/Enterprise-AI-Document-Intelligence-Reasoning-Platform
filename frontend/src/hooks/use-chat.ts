"use client"

import { useState, useCallback, useRef } from 'react'
import { ChatMessage, ChatSession } from '@/types'
import { ChatAPI } from '@/lib/api/chat'

interface UseChatOptions {
    sessionId?: string
    documentIds?: string[]
}

export function useChat(options: UseChatOptions = {}) {
    const { sessionId: initialSessionId, documentIds = [] } = options

    const [sessionId, setSessionId] = useState<string | null>(initialSessionId || null)
    const [messages, setMessages] = useState<ChatMessage[]>([])
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState<string | null>(null)
    const [isTyping, setIsTyping] = useState(false)
    const abortControllerRef = useRef<AbortController | null>(null)

    /**
     * Create a new chat session
     */
    const createSession = useCallback(async (title?: string, docIds?: string[]) => {
        const response = await ChatAPI.createSession(title, docIds || documentIds)

        if (response.success && response.data) {
            setSessionId(response.data.id)
            setMessages(response.data.messages || [])
            return { success: true, session: response.data }
        } else {
            setError(response.error?.message || 'Failed to create session')
            return { success: false, error: response.error?.message }
        }
    }, [documentIds])

    /**
     * Load an existing session
     */
    const loadSession = useCallback(async (id: string) => {
        setLoading(true)
        setError(null)

        const response = await ChatAPI.getSession(id)

        if (response.success && response.data) {
            setSessionId(response.data.id)
            setMessages(response.data.messages || [])
            setLoading(false)
            return { success: true, session: response.data }
        } else {
            setError(response.error?.message || 'Failed to load session')
            setLoading(false)
            return { success: false, error: response.error?.message }
        }
    }, [])

    /**
     * Send a message
     */
    const sendMessage = useCallback(
        async (content: string, docIds?: string[]) => {
            let currentSessionId = sessionId

            if (!currentSessionId) {
                // Auto-create session if not exists
                const createResult = await createSession(undefined, docIds)
                if (!createResult.success || !createResult.session) {
                    return { success: false, error: 'Failed to create session' }
                }
                currentSessionId = createResult.session.id
            }

            // Add user message optimistically
            const userMessage: ChatMessage = {
                id: `temp-${Date.now()}`,
                role: 'user',
                content,
                timestamp: new Date().toISOString(),
            }

            // Add empty assistant message placeholder
            const assistantMessageId = `ai-${Date.now()}`;
            const assistantMessagePlaceholder: ChatMessage = {
                id: assistantMessageId,
                role: 'assistant',
                content: '',
                timestamp: new Date().toISOString(),
            }

            setMessages((prev) => [...prev, userMessage, assistantMessagePlaceholder])
            setIsTyping(true)
            setError(null)

            try {
                await ChatAPI.streamMessage(
                    currentSessionId,
                    content,
                    (chunk) => {
                        if (chunk.type === 'start') {
                            setIsTyping(false); // Stop skeleton, start text
                        } else if (chunk.type === 'token') {
                            setMessages((prev) => {
                                const newMessages = [...prev];
                                const lastMsg = newMessages[newMessages.length - 1];
                                if (lastMsg.id === assistantMessageId) {
                                    lastMsg.content += chunk.content;
                                }
                                return newMessages;
                            });
                        } else if (chunk.type === 'done') {
                            // Finalize message with correct context
                            setMessages((prev) => {
                                const newMessages = [...prev];
                                const lastMsg = newMessages[newMessages.length - 1];
                                if (lastMsg.id === assistantMessageId) {
                                    lastMsg.content = chunk.content; // Ensure consistency
                                    if (chunk.document_context) {
                                        // Map backend snake_case to frontend camelCase
                                        lastMsg.documentContext = {
                                            documentId: chunk.document_context.document_id,
                                            documentName: chunk.document_context.document_name,
                                            relevantChunks: chunk.document_context.relevant_chunks
                                        };
                                    }
                                }
                                return newMessages;
                            });
                            setIsTyping(false);
                        } else if (chunk.type === 'status') {
                            setMessages((prev) => {
                                const newMessages = [...prev];
                                const lastMsg = newMessages[newMessages.length - 1];
                                if (lastMsg.id === assistantMessageId) {
                                    // Append status in italics
                                    const prefix = lastMsg.content ? '\n' : '';
                                    lastMsg.content += `${prefix}_${chunk.content}_`;
                                }
                                return newMessages;
                            });
                            setIsTyping(false);
                        } else if (chunk.type === 'error') {
                            setError(chunk.content);
                            setIsTyping(false);
                        }
                    }
                );
                return { success: true }
            } catch (err: any) {
                // Remove optimistic messages on error
                setMessages((prev) => prev.filter((m) => m.id !== userMessage.id && m.id !== assistantMessageId))
                setError(err.message || 'Failed to send message')
                setIsTyping(false)
                return { success: false, error: err.message }
            }
        },
        [sessionId, documentIds, createSession]
    )

    /**
     * Clear current session
     */
    const clearSession = useCallback(() => {
        setSessionId(null)
        setMessages([])
        setError(null)
    }, [])

    /**
     * Delete session
     */
    const deleteSession = useCallback(async (id?: string) => {
        const targetId = id || sessionId
        if (!targetId) return { success: false, error: 'No session to delete' }

        const response = await ChatAPI.deleteSession(targetId)

        if (response.success) {
            if (targetId === sessionId) {
                clearSession()
            }
            return { success: true }
        } else {
            setError(response.error?.message || 'Failed to delete session')
            return { success: false, error: response.error?.message }
        }
    }, [sessionId, clearSession])

    return {
        sessionId,
        messages,
        loading,
        error,
        isTyping,
        createSession,
        loadSession,
        sendMessage,
        clearSession,
        deleteSession,
    }
}
