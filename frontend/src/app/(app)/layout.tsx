/**
 * Shared layout for authenticated app pages (dashboard, knowledge-graph, etc.)
 *
 * This layout wraps all authenticated pages with a single SidebarProvider instance,
 * ensuring state persistence (chat sessions, sidebar data) across navigation.
 *
 * NOTE: This must NOT be "use client". Marking a layout as a Client Component causes
 * Radix UI's internal ID counter to diverge between SSR and client hydration,
 * producing React hydration errors. AppSidebar and SidebarProvider are already
 * "use client" — the layout itself should remain a Server Component.
 */

import React from 'react'
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
