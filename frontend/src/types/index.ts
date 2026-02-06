/**
 * Document model representing an uploaded file
 */
export interface Document {
    id: string
    filename: string
    originalName: string
    fileSize: number
    mimeType: string
    uploadedAt: string
    processedAt?: string
    status: 'uploading' | 'processing' | 'completed' | 'failed'
    metadata?: {
        pageCount?: number
        extractedText?: string
        vectorStoreId?: string
    }
}

/**
 * Chat message in a conversation
 */
export interface ChatMessage {
    id: string
    role: 'user' | 'assistant' | 'system'
    content: string
    timestamp: string
    reasoning?: string[]
    documentContext?: {
        documentId: string
        documentName: string
        relevantChunks?: string[]
    }
}

/**
 * Chat session/conversation
 */
export interface ChatSession {
    id: string
    title: string
    created_at: string
    updated_at: string
    messages: ChatMessage[]
    document_ids: string[]
}

/**
 * API Response wrapper
 */
export interface ApiResponse<T> {
    success: boolean
    data?: T
    error?: {
        message: string
        code: string
        details?: unknown
    }
}

/**
 * Pagination metadata
 */
export interface PaginationMeta {
    page: number
    pageSize: number
    total: number
    totalPages: number
}

/**
 * Paginated list response
 */
export interface PaginatedResponse<T> {
    items: T[]
    meta: PaginationMeta
}

/**
 * Document upload progress
 */
export interface UploadProgress {
    filename: string
    progress: number
    status: 'pending' | 'uploading' | 'processing' | 'completed' | 'error'
    error?: string
}
