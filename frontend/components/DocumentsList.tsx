export type Doc = {
  id: string;
  name: string;
};

export default function DocumentsList({
  documents,
  onRemove,
}: {
  documents: Doc[];
  onRemove: (id: string) => void;
}) {
  if (documents.length === 0) return null;

  return (
    <div className="w-full max-w-chat space-y-1.5">
      {documents.map((doc) => (
        <div
          key={doc.id}
          className="flex items-center justify-between rounded-md border border-slate/15 bg-white px-3 py-2 text-sm"
        >
          <span className="truncate text-slate">{doc.name}</span>
          <button
            onClick={() => onRemove(doc.id)}
            aria-label={`Remove ${doc.name}`}
            className="ml-2 shrink-0 text-slate/40 hover:text-citation"
          >
            <CloseIcon />
          </button>
        </div>
      ))}
    </div>
  );
}

function CloseIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
      <path d="M18 6L6 18M6 6l12 12" strokeLinecap="round" />
    </svg>
  );
}
