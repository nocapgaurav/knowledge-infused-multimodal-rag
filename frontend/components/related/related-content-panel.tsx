"use client";

import dynamic from "next/dynamic";
import { useState } from "react";
import { Loader2, Share2 } from "lucide-react";

import { EvidenceCard } from "@/components/evidence/evidence-card";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { TYPOGRAPHY } from "@/constants/typography";
import { useRelatedContent } from "@/services/queries";
import { EMPTY_TURNS, useConversationStore } from "@/store/conversation-store";
import { useWorkspaceStore } from "@/store/workspace-store";
import type { ChunkModality, RetrievalCandidateDto } from "@/types/api";

/** React Flow is a sizeable dependency only ever needed once a user
 * explicitly opens the Graph tab (Phase 3: relationship visualization is
 * optional and user-initiated) -- deferring its import keeps it out of
 * every other page's bundle. */
const RelationshipGraph = dynamic(
  () =>
    import("@/components/related/relationship-graph").then((module) => module.RelationshipGraph),
  {
    loading: () => (
      <div className="flex h-full items-center justify-center">
        <Loader2 className="text-muted-foreground size-6 animate-spin" />
      </div>
    ),
  },
);

/**
 * Related Content (Phase 3/4B): relationships already discovered by
 * Module 9's graph expansion for the most recent question -- never
 * speculative. Optional and user-initiated; evidence, not this panel,
 * is the default trust mechanism. Offers both a List view (the default,
 * consistent with every other panel's card-based reading experience) and
 * a Graph view (Phase 3's named "Relationship Viewer", via React Flow).
 */
export function RelatedContentPanel({ documentId }: { documentId: string }) {
  const [explored, setExplored] = useState(false);
  const turns = useConversationStore((state) => state.turnsByDocument[documentId] ?? EMPTY_TURNS);
  const openEvidence = useWorkspaceStore((state) => state.openEvidence);
  const openedEvidenceId = useWorkspaceStore((state) => state.openedEvidenceId);

  const latestQuestion =
    [...turns].reverse().find((turn) => turn.status === "complete")?.question ?? null;
  const { data, isPending, isError } = useRelatedContent(
    documentId,
    explored ? latestQuestion : null,
  );

  if (!latestQuestion) {
    return <EmptyState message="Related content becomes available once you've asked a question." />;
  }

  if (!explored) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-3 p-6 text-center">
        <Share2 className="text-muted-foreground size-6" aria-hidden="true" />
        <p className={TYPOGRAPHY.body}>
          Explore how this answer connects to the rest of the paper.
        </p>
        <Button variant="outline" size="sm" onClick={() => setExplored(true)}>
          Explore related content
        </Button>
      </div>
    );
  }

  if (isPending) {
    return (
      <div className="flex h-full items-center justify-center">
        <Loader2 className="text-muted-foreground size-6 animate-spin" />
      </div>
    );
  }

  if (isError || !data) {
    return <EmptyState message="Could not load related content right now." />;
  }

  const relatedCandidates = data.bundle.candidates.filter(
    (candidate) => candidate.discovery_method === "graph_expansion",
  );

  if (relatedCandidates.length === 0) {
    return <EmptyState message="No related content was found for this question." />;
  }

  return (
    <Tabs defaultValue="list" className="flex h-full flex-col gap-0">
      <TabsList className="m-2 w-fit">
        <TabsTrigger value="list">List</TabsTrigger>
        <TabsTrigger value="graph">Graph</TabsTrigger>
      </TabsList>
      <TabsContent value="list" className="min-h-0 flex-1">
        <ScrollArea className="h-full">
          <div className="flex flex-col gap-2 p-3">
            {relatedCandidates.map((candidate) => (
              <EvidenceCard
                key={candidate.knowledge_unit_id}
                item={{
                  knowledgeUnitId: candidate.knowledge_unit_id,
                  label: labelFor(candidate.modality),
                  displayLabel: candidate.retrieval_context ?? undefined,
                  pageNumbers: candidate.page_numbers?.length ? candidate.page_numbers : undefined,
                  discovery: discoveryFor(candidate),
                  modality: candidate.modality,
                  text: candidate.text,
                  groupId: candidate.knowledge_unit_id,
                  isPrimary: false,
                }}
                isActive={candidate.knowledge_unit_id === openedEvidenceId}
                onSelect={() => openEvidence(candidate.knowledge_unit_id)}
              />
            ))}
          </div>
        </ScrollArea>
      </TabsContent>
      <TabsContent value="graph" className="min-h-0 flex-1">
        <RelationshipGraph candidates={data.bundle.candidates} />
      </TabsContent>
    </Tabs>
  );
}

function labelFor(modality: ChunkModality): string {
  return modality === "figure" ? "Figure" : modality === "table" ? "Table" : "Related";
}

/** Honest provenance for a related item, from its real graph path: which
 * relationship connected it to the evidence that answered the question. */
function discoveryFor(candidate: RetrievalCandidateDto): string | undefined {
  const lastHop = candidate.graph_path.hops.at(-1);
  if (!lastHop) return undefined;
  const relationship = lastHop.relationship_type;
  const REASON: Record<string, string> = {
    CITES: "Cited by matched evidence",
    REFERENCES: "Referenced by matched evidence",
    CONTINUES: "Continues a matched passage",
    NEXT: "Adjacent to matched evidence",
    BELONGS_TO: "Same section as matched evidence",
  };
  return REASON[relationship] ?? `Connected via ${relationship.toLowerCase()}`;
}

function EmptyState({ message }: { message: string }) {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-2 p-6 text-center">
      <Share2 className="text-muted-foreground size-6" aria-hidden="true" />
      <p className={TYPOGRAPHY.body}>{message}</p>
    </div>
  );
}
