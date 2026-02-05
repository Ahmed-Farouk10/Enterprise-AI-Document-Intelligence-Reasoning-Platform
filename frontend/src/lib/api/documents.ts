import { ApiClient } from '../api-client'
import { Document, PaginatedResponse, UploadProgress } from '@/types'

/**
 * Document API endpoints
 */
export const DocumentAPI = {
    /**
     * Get all documents with pagination
     */
    async getDocuments(page = 1, pageSize = 20) {
        return ApiClient.get<PaginatedResponse<Document>>('/api/documents', {
            params: { page, page_size: pageSize },
        })
    },

    /**
     * Get a single document by ID
     */
    async getDocument(id: string) {
        return ApiClient.get<Document>(`/api/documents/${id}`)
    },

    /**
     * Upload a new document
     */
    async uploadDocument(file: File, onProgress?: (progress: number) => void) {
        return ApiClient.uploadFile<Document>('/api/documents/upload', file, onProgress)
    },

    /**
     * Delete a document
     */
    async deleteDocument(id: string) {
        return ApiClient.delete<void>(`/api/documents/${id}`)
    },

    /**
     * Get document processing status
     */
    async getDocumentStatus(id: string) {
        return ApiClient.get<{ status: string; progress?: number }>(`/api/documents/${id}/status`)
    },
}
