import { useState } from "react";

import ChestIcon from "./components/icons/ChestIcon";
import ImageUpload from "./components/uploader/ImageUploader";
import ModelReport from "./components/report/ModelReport";

type View = "score" | "report";

const TABS: { id: View; label: string }[] = [
  { id: "score", label: "Score an image" },
  { id: "report", label: "What these models are" },
];

function App() {
  const [view, setView] = useState<View>("score");

  return (
    // flex-col + min-h-screen, with <main> as the only flex-1 child: on a
    // short page main grows to fill the remaining height and the footer is
    // pushed to the bottom of the viewport; on a tall page the flex column
    // simply grows past 100vh and everything scrolls normally.
    <div className="flex min-h-screen w-full flex-col bg-slate-100">
      <header className="border-b border-slate-300 bg-white">
        <div className="w-full px-6 py-4">
          <div className="flex items-center gap-3">
            <ChestIcon className="h-8 w-8 shrink-0 text-slate-700" />
            <h1 className="text-xl font-semibold text-slate-900">
              Chest radiograph classifier
            </h1>
          </div>
          <p className="mt-1 text-sm text-slate-600">
            Two models, and an honest account of what each one is reading.
          </p>

          <nav className="mt-4 flex gap-2" aria-label="Views">
            {TABS.map((tab) => (
              <button
                key={tab.id}
                type="button"
                onClick={() => setView(tab.id)}
                aria-current={view === tab.id ? "page" : undefined}
                className={`rounded px-3 py-1.5 text-sm ${
                  view === tab.id
                    ? "bg-slate-800 text-white"
                    : "bg-slate-200 text-slate-800 hover:bg-slate-300"
                }`}
              >
                {tab.label}
              </button>
            ))}
          </nav>
        </div>
      </header>

      <main className="w-full flex-1 px-6 py-6">
        {view === "score" ? (
          <ImageUpload />
        ) : (
          <div className="rounded-lg bg-white p-6 shadow-sm">
            <ModelReport />
          </div>
        )}
      </main>

      <footer className="border-t border-slate-300 bg-white">
        <div className="w-full px-4 py-4 text-sm text-slate-600 text-center">
          Research artefact built to measure a documented shortcut-learning failure mode.
          Not a diagnostic device, and not usable as one.
        </div>
      </footer>
    </div>
  );
}

export default App;
