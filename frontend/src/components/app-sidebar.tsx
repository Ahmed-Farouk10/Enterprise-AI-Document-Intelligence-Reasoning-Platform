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

const data = {
    teams: [
        {
            name: 'DocuCentric',
            logo: '📄',
            plan: 'Enterprise',
        },
    ],
    chats: [
        {
            name: 'Q&A about Contract.pdf',
            url: '#',
            date: '2 hours ago',
        },
        {
            name: 'Research Summary',
            url: '#',
            date: 'Yesterday',
        },
    ],
}

export function AppSidebar({ ...props }: React.ComponentProps<typeof Sidebar>) {
    const { documents, loading, uploadDocument, deleteDocument, uploadProgress, error } = useDocuments()
    const { toast } = useToast()
    const [isUploading, setIsUploading] = React.useState(false)
    const [currentProgress, setCurrentProgress] = React.useState(0)
    const [uploadError, setUploadError] = React.useState<string | null>(null)

    const handleUpload = async (file: File) => {
        setIsUploading(true)
        setUploadError(null)
        setCurrentProgress(0)

        const result = await uploadDocument(file)

        setIsUploading(false)
        setCurrentProgress(0)

        if (result.success) {
            toast({
                title: "Upload Successful",
                description: `${file.name} has been uploaded successfully.`,
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
        url: `http://localhost:8000/api/documents/${doc.id}/download`,
        date: new Date(doc.uploadedAt).toLocaleDateString(),
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
                <NavHistory chats={data.chats} />
                <NavSystemStatus />
            </SidebarContent>
            <SidebarRail />
        </Sidebar>
    )
}
