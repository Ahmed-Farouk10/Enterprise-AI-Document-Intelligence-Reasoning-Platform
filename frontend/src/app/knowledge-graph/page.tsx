"use client"

import { AppSidebar } from '@/components/app-sidebar'
import {
    Breadcrumb,
    BreadcrumbItem,
    BreadcrumbLink,
    BreadcrumbList,
    BreadcrumbPage,
    BreadcrumbSeparator,
} from '@/components/ui/breadcrumb'
import { Separator } from '@/components/ui/separator'
import {
    SidebarInset,
    SidebarProvider,
    SidebarTrigger,
} from '@/components/ui/sidebar'
import { Button } from "@/components/ui/button"
import { Network, Download, ZoomIn, ZoomOut, Maximize2, RefreshCw } from "lucide-react"
import { KnowledgeGraphVisualization } from '@/components/knowledge-graph-visualization'
import { GraphStats } from '@/components/graph-stats'
import { useState } from 'react'

export default function KnowledgeGraphPage() {
    const [isRefreshing, setIsRefreshing] = useState(false)

    const handleRefresh = async () => {
        setIsRefreshing(true)
        // TODO: Implement refresh logic
        setTimeout(() => setIsRefreshing(false), 1000)
    }

    const handleExport = () => {
        // TODO: Implement export logic
        console.log('Exporting graph...')
    }

    return (
        <SidebarProvider>
            <AppSidebar />
            <SidebarInset>
                <header className="flex h-16 shrink-0 items-center gap-2 transition-[width,height] ease-linear group-has-data-[collapsible=icon]/sidebar-wrapper:h-12">
                    <div className="flex w-full items-center justify-between px-4">
                        <div className="flex items-center gap-2">
                            <SidebarTrigger className="-ml-1" />
                            <Separator
                                orientation="vertical"
                                className="mr-2 data-[orientation=vertical]:h-4"
                            />
                            <Breadcrumb>
                                <BreadcrumbList>
                                    <BreadcrumbItem className="hidden md:block">
                                        <BreadcrumbLink href="#">
                                            DocuCentric
                                        </BreadcrumbLink>
                                    </BreadcrumbItem>
                                    <BreadcrumbSeparator className="hidden md:block" />
                                    <BreadcrumbItem>
                                        <BreadcrumbPage>Knowledge Graph</BreadcrumbPage>
                                    </BreadcrumbItem>
                                </BreadcrumbList>
                            </Breadcrumb>
                        </div>
                        <div className="flex items-center gap-2">
                            <Button
                                variant="outline"
                                size="sm"
                                onClick={handleRefresh}
                                disabled={isRefreshing}
                            >
                                <RefreshCw className={`mr-2 size-4 ${isRefreshing ? 'animate-spin' : ''}`} />
                                Refresh
                            </Button>
                            <Button
                                variant="outline"
                                size="sm"
                                onClick={handleExport}
                            >
                                <Download className="mr-2 size-4" />
                                Export
                            </Button>
                        </div>
                    </div>
                </header>
                <div className="flex flex-1 flex-col gap-4 overflow-hidden p-4">
                    {/* Graph Statistics */}
                    <GraphStats />

                    {/* Main Graph Visualization */}
                    <div className="flex-1 overflow-hidden rounded-xl border bg-card shadow-sm">
                        <KnowledgeGraphVisualization />
                    </div>
                </div>
            </SidebarInset>
        </SidebarProvider>
    )
}
