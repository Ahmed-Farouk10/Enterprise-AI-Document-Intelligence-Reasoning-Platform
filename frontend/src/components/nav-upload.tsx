"use client"

import { CloudUpload, Loader2, CheckCircle2, AlertCircle } from "lucide-react"
import { useRef, useState } from "react"
import {
    SidebarGroup,
    SidebarGroupLabel,
    SidebarMenu,
    SidebarMenuItem,
} from "@/components/ui/sidebar"

interface NavUploadProps {
    onUpload?: (file: File) => Promise<void>
    uploading?: boolean
    uploadProgress?: number
    uploadError?: string | null
}

export function NavUpload({ onUpload, uploading = false, uploadProgress = 0, uploadError = null }: NavUploadProps) {
    const fileInputRef = useRef<HTMLInputElement>(null)
    const [isDragging, setIsDragging] = useState(false)

    const handleFileSelect = async (file: File) => {
        // Validation
        try {
            const { FileUploadSchema } = require("@/lib/validation")
            FileUploadSchema.parse({ file })
        } catch (err: any) {
            const { toast } = require("@/hooks/use-toast")
            toast({
                title: "Invalid File",
                description: err.errors?.[0]?.message || "Invalid file",
                variant: "destructive"
            })
            if (fileInputRef.current) fileInputRef.current.value = ""
            return
        }

        if (onUpload) {
            await onUpload(file)
        }
    }

    const handleClick = () => {
        fileInputRef.current?.click()
    }

    const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0]
        if (file) {
            handleFileSelect(file)
        }
    }

    const handleDragOver = (e: React.DragEvent) => {
        e.preventDefault()
        setIsDragging(true)
    }

    const handleDragLeave = () => {
        setIsDragging(false)
    }

    const handleDrop = async (e: React.DragEvent) => {
        e.preventDefault()
        setIsDragging(false)

        const file = e.dataTransfer.files?.[0]
        if (file) {
            handleFileSelect(file)
        }
    }

    return (
        <SidebarGroup className="group-data-[collapsible=icon]:hidden">
            <SidebarGroupLabel>Upload Content</SidebarGroupLabel>
            <SidebarMenu>
                <SidebarMenuItem>
                    <input
                        ref={fileInputRef}
                        type="file"
                        accept=".pdf,.txt,.docx,.doc"
                        onChange={handleFileInputChange}
                        className="hidden"
                    />
                    <button
                        onClick={handleClick}
                        onDragOver={handleDragOver}
                        onDragLeave={handleDragLeave}
                        onDrop={handleDrop}
                        disabled={uploading}
                        className={`flex h-32 w-full flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed p-4 text-sidebar-foreground transition-all ${isDragging
                            ? 'border-primary bg-primary/10'
                            : 'border-sidebar-border bg-sidebar-accent/30 hover:bg-sidebar-accent hover:text-sidebar-accent-foreground'
                            } ${uploading ? 'cursor-not-allowed opacity-60' : 'cursor-pointer'
                            } focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sidebar-ring`}
                    >
                        {uploading ? (
                            <>
                                <Loader2 className="size-8 animate-spin text-muted-foreground/80" />
                                <div className="flex flex-col gap-1 text-center text-xs">
                                    <span className="font-medium">Uploading...</span>
                                    <span className="text-muted-foreground">{uploadProgress}%</span>
                                </div>
                            </>
                        ) : uploadError ? (
                            <>
                                <AlertCircle className="size-8 text-destructive" />
                                <div className="flex flex-col gap-1 text-center text-xs">
                                    <span className="font-medium text-destructive">Upload Failed</span>
                                    <span className="text-muted-foreground text-[10px]">{uploadError}</span>
                                </div>
                            </>
                        ) : (
                            <>
                                <CloudUpload className="size-8 text-muted-foreground/80" />
                                <div className="flex flex-col gap-1 text-center text-xs">
                                    <span className="font-medium">Upload Documents</span>
                                    <span className="text-muted-foreground">PDF, TXT, or DOCX</span>
                                </div>
                            </>
                        )}
                    </button>
                </SidebarMenuItem>
            </SidebarMenu>
        </SidebarGroup>
    )
}
