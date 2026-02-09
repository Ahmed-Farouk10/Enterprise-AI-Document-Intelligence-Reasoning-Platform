"use client"

import { Card } from "@/components/ui/card"
import { Network, GitBranch, FileText, Link2 } from "lucide-react"
import { useEffect, useState } from "react"

interface GraphStatsData {
    totalEntities: number
    totalRelationships: number
    totalDocuments: number
    graphDensity: number
}

export function GraphStats() {
    const [stats, setStats] = useState<GraphStatsData>({
        totalEntities: 0,
        totalRelationships: 0,
        totalDocuments: 0,
        graphDensity: 0
    })
    const [isLoading, setIsLoading] = useState(true)

    useEffect(() => {
        const fetchStats = async () => {
            try {
                const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
                const response = await fetch(`${apiUrl}/api/graph/stats`)

                if (!response.ok) {
                    throw new Error('Failed to fetch graph stats')
                }

                const data = await response.json()

                setStats({
                    totalEntities: data.total_entities || 0,
                    totalRelationships: data.total_relationships || 0,
                    totalDocuments: data.total_documents || 0,
                    graphDensity: data.graph_density || 0
                })
                setIsLoading(false)
            } catch (error) {
                console.error('Failed to fetch graph stats:', error)
                setIsLoading(false)
            }
        }

        fetchStats()
    }, [])

    const statCards = [
        {
            title: "Entities",
            value: stats.totalEntities,
            icon: Network,
            description: "Total nodes in graph",
            color: "text-blue-500"
        },
        {
            title: "Relationships",
            value: stats.totalRelationships,
            icon: GitBranch,
            description: "Total connections",
            color: "text-green-500"
        },
        {
            title: "Documents",
            value: stats.totalDocuments,
            icon: FileText,
            description: "Processed documents",
            color: "text-purple-500"
        },
        {
            title: "Graph Density",
            value: `${(stats.graphDensity * 100).toFixed(0)}%`,
            icon: Link2,
            description: "Connection ratio",
            color: "text-orange-500"
        }
    ]

    return (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            {statCards.map((stat) => (
                <Card key={stat.title} className="p-6">
                    <div className="flex items-center justify-between">
                        <div className="space-y-1">
                            <p className="text-sm font-medium text-muted-foreground">
                                {stat.title}
                            </p>
                            <p className="text-2xl font-bold">
                                {isLoading ? (
                                    <span className="inline-block h-8 w-16 animate-pulse rounded bg-muted" />
                                ) : (
                                    stat.value
                                )}
                            </p>
                            <p className="text-xs text-muted-foreground">
                                {stat.description}
                            </p>
                        </div>
                        <div className={`rounded-full bg-muted p-3 ${stat.color}`}>
                            <stat.icon className="size-5" />
                        </div>
                    </div>
                </Card>
            ))}
        </div>
    )
}
