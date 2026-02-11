/**
 * Shared layout for authenticated app pages (dashboard, knowledge-graph, etc.)
 * 
 * This layout wraps all authenticated pages with a single SidebarProvider instance,
 * ensuring state persistence (chat sessions, sidebar data) across navigation.
 */

"use client"

import { AppSidebar } from '@/components/app-sidebar'
import { SidebarInset, SidebarProvider } from '@/components/ui/sidebar'

export default function AppLayout({
    children,
}: {
    children: React.ReactNode
}) {
    return (
        <SidebarProvider>
            <AppSidebar />
            <SidebarInset>
                {children}
            </SidebarInset>
        </SidebarProvider>
    )
}
