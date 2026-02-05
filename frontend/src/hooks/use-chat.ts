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

            setMessages((prev) => [...prev, userMessage])
            setIsTyping(true)
            setError(null)

            const response = await ChatAPI.sendMessage(currentSessionId, content, docIds || documentIds)

            setIsTyping(false)

            if (response.success && response.data) {
                // Just add the assistant response (user message already added optimistically)
                setMessages((prev) => [...prev, response.data!])
                return { success: true, message: response.data }
            } else {
                // Remove optimistic user message on error
                setMessages((prev) => prev.filter((m) => m.id !== userMessage.id))
                setError(response.error?.message || 'Failed to send message')
                return { success: false, error: response.error?.message }
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
