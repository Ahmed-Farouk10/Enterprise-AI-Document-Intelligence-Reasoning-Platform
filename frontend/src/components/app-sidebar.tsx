"use client"

import * as React from "react"
import { NavUpload } from '@/components/nav-upload'
import { NavDocuments } from '@/components/nav-documents'
import { NavHistory } from '@/components/nav-history'
import { NavSystemStatus } from '@/components/nav-system-status'
import { TeamSwitcher } from '@/components/team-switcher'
import {
    Sidebar,
    SidebarContent,
    SidebarHeader,
    SidebarRail,
} from '@/components/ui/sidebar'
import { useDocuments } from '@/hooks/use-documents'
import { useToast } from '@/hooks/use-toast'

import { ChatAPI } from '@/lib/api/chat'
import { ChatSession } from '@/types'

const data = {
    teams: [
        {
            name: 'DocuCentric',
            logo: '📄',
            plan: 'Enterprise',
        },
    ],
    // Removed mock chats
}

export function AppSidebar({ ...props }: React.ComponentProps<typeof Sidebar>) {
    const { documents, loading, uploadDocument, deleteDocument, uploadProgress, error, fetchDocuments } = useDocuments()
    const { toast } = useToast()
    const [isUploading, setIsUploading] = React.useState(false)
    const [currentProgress, setCurrentProgress] = React.useState(0)
    const [uploadError, setUploadError] = React.useState<string | null>(null)
    const [chatSessions, setChatSessions] = React.useState<ChatSession[]>([])
    const [sessionsLoading, setSessionsLoading] = React.useState(true)
    const processedDocIds = React.useRef<Set<string>>(new Set())

    // Initial setup for processed IDs (to avoid re-toasting old completed docs on refresh)
    React.useEffect(() => {
        if (documents.length > 0 && processedDocIds.current.size === 0) {
            documents.forEach(doc => {
                if (doc.status === 'completed') processedDocIds.current.add(doc.id)
            })
        }
    }, [documents])

    // Handle document completion notifications
    React.useEffect(() => {
        documents.forEach(doc => {
            if (doc.status === 'completed' && !processedDocIds.current.has(doc.id)) {
                processedDocIds.current.add(doc.id)

                // Extract details from metadata
                const stats = (doc.metadata as any)?.graph_stats

                // Construct the message the user requested:
                // ✅ Resume processed: Ahmed Ayman Farouk Shahin, 5 positions, 0 skills
                const personName = stats?.entities?.Person?.[0]?.name || doc.originalName || 'Document'
                const positionCount = stats?.graph_stats?.entities?.Position?.length || stats?.entities?.Position?.length || 0
                const skillCount = stats?.graph_stats?.entities?.Skill?.length || stats?.entities?.Skill?.length || 0

                const detailStr = stats
                    ? `✅ Resume processed: ${personName}, ${positionCount} positions, ${skillCount} skills`
                    : `✅ Document processed successfully: ${doc.originalName}`

                toast({
                    title: "Processing Complete",
                    description: detailStr,
                })
            }
        })
    }, [documents, toast])

    // Poll for updates (Chat sessions & Documents)
    React.useEffect(() => {
        const fetchSessions = async () => {
            try {
                const response = await ChatAPI.getSessions()
                if (response.success && response.data) {
                    setChatSessions(response.data)
                }
            } catch (err) {
                console.error("Failed to fetch chat history", err)
            } finally {
                setSessionsLoading(false)
            }
        }

        // Initial fetch
        fetchSessions()

        // Poll every 5 seconds to keep UI in sync with background processing
        const interval = setInterval(() => {
            fetchSessions()
            // Refresh documents list to catch status changes (pending -> completed)
            if (fetchDocuments) {
                fetchDocuments()
            }
        }, 5000)

        return () => clearInterval(interval)
    }, [fetchDocuments])

    const handleUpload = async (file: File) => {
        setIsUploading(true)
        setUploadError(null)
        setCurrentProgress(0)

        const result = await uploadDocument(file)

        setIsUploading(false)
        setCurrentProgress(0)

        if (result.success) {
            toast({
                title: "Upload Started",
                description: `${file.name} is being processed in the background.`,
            })
        } else {
            setUploadError(result.error || 'Upload failed')
            toast({
                title: "Upload Failed",
                description: result.error || 'An error occurred during upload',
                variant: "destructive",
            })
        }
    }

    const handleDelete = async (id: string) => {
        const result = await deleteDocument(id)

        if (result.success) {
            toast({
                title: "Document Deleted",
                description: "The document has been removed.",
            })
        } else {
            toast({
                title: "Delete Failed",
                description: result.error || 'Failed to delete document',
                variant: "destructive",
            })
        }
    }

    // Map documents to the format expected by NavDocuments
    const mappedDocuments = documents.map((doc) => ({
        id: doc.id,
        name: doc.originalName || doc.filename,
        url: `${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/documents/${doc.id}/download`,
        date: new Date(doc.uploadedAt).toLocaleDateString(),
    }))

    // Map chat sessions for NavHistory
    const mappedChats = chatSessions.map(session => ({
        name: session.title || 'New Chat',
        url: '#', // TODO: Add routing to specific chat
        date: new Date(session.updated_at || session.created_at).toLocaleDateString()
    }))

    // Track upload progress
    React.useEffect(() => {
        if (Object.keys(uploadProgress).length > 0) {
            const progress = Object.values(uploadProgress)[0] || 0
            setCurrentProgress(progress)
        }
    }, [uploadProgress])

    return (
        <Sidebar collapsible="icon" {...props}>
            <SidebarHeader>
                <TeamSwitcher teams={data.teams} />
            </SidebarHeader>
            <SidebarContent>
                <NavUpload
                    onUpload={handleUpload}
                    uploading={isUploading}
                    uploadProgress={currentProgress}
                    uploadError={uploadError}
                />
                <NavDocuments
                    documents={mappedDocuments}
                    loading={loading}
                    onDelete={handleDelete}
                />
                <NavHistory chats={mappedChats} loading={sessionsLoading} />
                <NavSystemStatus />
            </SidebarContent>
            <SidebarRail />
        </Sidebar>
    )
}
