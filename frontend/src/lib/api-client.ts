import axios, { AxiosInstance, AxiosError, AxiosRequestConfig } from 'axios'
import { ApiResponse } from '@/types'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

/**
 * Axios instance with default configuration
 */
const axiosInstance: AxiosInstance = axios.create({
    baseURL: API_BASE_URL,
    timeout: 30000,
    headers: {
        'Content-Type': 'application/json',
    },
})

/**
 * Request interceptor for adding auth tokens, logging, etc.
 */
axiosInstance.interceptors.request.use(
    (config) => {
        // Add auth token if available
        const token = localStorage.getItem('auth_token')
        if (token) {
            config.headers.Authorization = `Bearer ${token}`
        }
        return config
    },
    (error) => {
        return Promise.reject(error)
    }
)

/**
 * Response interceptor for handling errors globally
 */
axiosInstance.interceptors.response.use(
    (response) => response,
    (error: AxiosError) => {
        // Handle common errors
        if (error.response?.status === 401) {
            // Redirect to login or refresh token
            console.error('Unauthorized - redirecting to login')
        }
        return Promise.reject(error)
    }
)

/**
 * Generic API client wrapper
 */
export class ApiClient {
    /**
     * GET request
     */
    static async get<T>(url: string, config?: AxiosRequestConfig): Promise<ApiResponse<T>> {
        try {
            const response = await axiosInstance.get<T>(url, config)
            return {
                success: true,
                data: response.data,
            }
        } catch (error) {
            return this.handleError(error)
        }
    }

    /**
     * POST request
     */
    static async post<T>(
        url: string,
        data?: unknown,
        config?: AxiosRequestConfig
    ): Promise<ApiResponse<T>> {
        try {
            const response = await axiosInstance.post<T>(url, data, config)
            return {
                success: true,
                data: response.data,
            }
        } catch (error) {
            return this.handleError(error)
        }
    }

    /**
     * PUT request
     */
    static async put<T>(
        url: string,
        data?: unknown,
        config?: AxiosRequestConfig
    ): Promise<ApiResponse<T>> {
        try {
            const response = await axiosInstance.put<T>(url, data, config)
            return {
                success: true,
                data: response.data,
            }
        } catch (error) {
            return this.handleError(error)
        }
    }

    /**
     * DELETE request
     */
    static async delete<T>(url: string, config?: AxiosRequestConfig): Promise<ApiResponse<T>> {
        try {
            const response = await axiosInstance.delete<T>(url, config)
            return {
                success: true,
                data: response.data,
            }
        } catch (error) {
            return this.handleError(error)
        }
    }

    /**
     * Upload file with progress tracking
     */
    static async uploadFile<T>(
        url: string,
        file: File,
        onProgress?: (progress: number) => void
    ): Promise<ApiResponse<T>> {
        try {
            const formData = new FormData()
            formData.append('file', file)

            const response = await axiosInstance.post<T>(url, formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
                onUploadProgress: (progressEvent) => {
                    if (progressEvent.total && onProgress) {
                        const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total)
                        onProgress(percentCompleted)
                    }
                },
            })

            return {
                success: true,
                data: response.data,
            }
        } catch (error) {
            return this.handleError(error)
        }
    }

    /**
     * Error handler
     */
    private static handleError(error: unknown): ApiResponse<never> {
        if (axios.isAxiosError(error)) {
            const axiosError = error as AxiosError<{ message?: string; detail?: string }>
            return {
                success: false,
                error: {
                    message: axiosError.response?.data?.message || axiosError.response?.data?.detail || axiosError.message || 'An error occurred',
                    code: axiosError.code || 'UNKNOWN_ERROR',
                    details: axiosError.response?.data,
                },
            }
        }

        return {
            success: false,
            error: {
                message: 'An unexpected error occurred',
                code: 'UNEXPECTED_ERROR',
            },
        }
    }
}

export default axiosInstance
