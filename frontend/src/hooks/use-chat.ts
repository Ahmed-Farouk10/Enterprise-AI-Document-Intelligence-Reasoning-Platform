"use client"

import { useState, useCallback, useRef, useEffect } from 'react'
import { ChatMessage, ChatSession } from '@/types'
import { ChatAPI } from '@/lib/api/chat'

const SESSION_STORAGE_KEY = 'docucentric_chat_session_id'

interface UseChatOptions {
    sessionId?: string
    documentIds?: string[]
}

export function useChat(options: UseChatOptions = {}) {
    const { sessionId: initialSessionId, documentIds = [] } = options

    // Restore sessionId from sessionStorage on first mount
    const [sessionId, _setSessionId] = useState<string | null>(() => {
        if (initialSessionId) return initialSessionId
        if (typeof window !== 'undefined') {
            return sessionStorage.getItem(SESSION_STORAGE_KEY) || null
        }
        return null
    })

    // Wrapper that also persists to sessionStorage
    const setSessionId = useCallback((id: string | null) => {
        _setSessionId(id)
        if (typeof window !== 'undefined') {
            if (id) {
                sessionStorage.setItem(SESSION_STORAGE_KEY, id)
            } else {
                sessionStorage.removeItem(SESSION_STORAGE_KEY)
            }
        }
    }, [])

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
            } else if (docIds && docIds.length > 0) {
                // If session exists but document selection changed, update it
                // This ensures multi-document retrieval is updated on-the-fly
                await ChatAPI.updateSession(currentSessionId, docIds)
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
                reasoning: [], // Initialize reasoning array
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
                                    lastMsg.content += chunk.data;
                                }
                                return newMessages;
                            });
                        } else if (chunk.type === 'done') {
                            // Finalize message with correct context
                            setMessages((prev) => {
                                const newMessages = [...prev];
                                const lastMsg = newMessages[newMessages.length - 1];
                                if (lastMsg.id === assistantMessageId) {
                                    lastMsg.content = chunk.data; // Ensure consistency
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
                                    // Update Reasoning - Immutable Update
                                    const newReasoning = [...(lastMsg.reasoning || []), chunk.data];

                                    newMessages[newMessages.length - 1] = {
                                        ...lastMsg,
                                        reasoning: newReasoning
                                    };
                                }
                                return newMessages;
                            });
                            setIsTyping(false);
                        } else if (chunk.type === 'error') {
                            setError(chunk.data);
                            setIsTyping(false);
                        }
                    }
                );
                return { success: true }
            } catch (err: any) {
                // If 404 (Session Not Found), clear session and retry ONCE
                if (err.message === 'Not Found' || err.status === 404) {
                    console.warn("Session not found (404), clearing stale session and retrying...");
                    clearSession();
                    // Remove the placeholder assistant message before retrying
                    setMessages((prev) => prev.filter((m) => m.id !== assistantMessageId));
                    return sendMessage(content, docIds);
                }

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
        setSessionId(null)   // also clears sessionStorage via wrapper
        setMessages([])
        setError(null)
    }, [setSessionId])

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
