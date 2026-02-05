"use client"

import { useState, useEffect, useCallback } from 'react'
import { Document, ApiResponse } from '@/types'
import { DocumentAPI } from '@/lib/api/documents'

interface UseDocumentsOptions {
    autoFetch?: boolean
    page?: number
    pageSize?: number
}

export function useDocuments(options: UseDocumentsOptions = {}) {
    const { autoFetch = true, page = 1, pageSize = 20 } = options

    const [documents, setDocuments] = useState<Document[]>([])
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState<string | null>(null)
    const [uploadProgress, setUploadProgress] = useState<Record<string, number>>({})

    /**
     * Fetch all documents
     */
    const fetchDocuments = useCallback(async () => {
        setLoading(true)
        setError(null)

        const response = await DocumentAPI.getDocuments(page, pageSize)

        if (response.success && response.data) {
            setDocuments(response.data.items)
        } else {
            setError(response.error?.message || 'Failed to fetch documents')
        }

        setLoading(false)
    }, [page, pageSize])

    /**
     * Upload a document
     */
    const uploadDocument = useCallback(async (file: File) => {
        const tempId = `temp-${Date.now()}`
        setUploadProgress((prev) => ({ ...prev, [tempId]: 0 }))

        const response = await DocumentAPI.uploadDocument(file, (progress) => {
            setUploadProgress((prev) => ({ ...prev, [tempId]: progress }))
        })

        setUploadProgress((prev) => {
            const updated = { ...prev }
            delete updated[tempId]
            return updated
        })

        if (response.success && response.data) {
            setDocuments((prev) => [response.data!, ...prev])
            return { success: true, document: response.data }
        } else {
            setError(response.error?.message || 'Failed to upload document')
            return { success: false, error: response.error?.message }
        }
    }, [])

    /**
     * Delete a document
     */
    const deleteDocument = useCallback(async (id: string) => {
        const response = await DocumentAPI.deleteDocument(id)

        if (response.success) {
            setDocuments((prev) => prev.filter((doc) => doc.id !== id))
            return { success: true }
        } else {
            setError(response.error?.message || 'Failed to delete document')
            return { success: false, error: response.error?.message }
        }
    }, [])

    /**
     * Get a single document
     */
    const getDocument = useCallback(async (id: string) => {
        const response = await DocumentAPI.getDocument(id)

        if (response.success && response.data) {
            return { success: true, document: response.data }
        } else {
            setError(response.error?.message || 'Failed to fetch document')
            return { success: false, error: response.error?.message }
        }
    }, [])

    /**
     * Auto-fetch on mount if enabled
     */
    useEffect(() => {
        if (autoFetch) {
            fetchDocuments()
        }
    }, [autoFetch, fetchDocuments])

    return {
        documents,
        loading,
        error,
        uploadProgress,
        fetchDocuments,
        uploadDocument,
        deleteDocument,
        getDocument,
    }
}
