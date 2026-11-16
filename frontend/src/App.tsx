import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { AppLayout } from './components/layout/AppLayout';
import { OverviewPage } from './pages/OverviewPage';
import { IngestPage } from './pages/IngestPage';
import { MemoriesPage } from './pages/MemoriesPage';
import { GraphPage } from './pages/GraphPage';
import { TimelinePage } from './pages/TimelinePage';
import { AskPage } from './pages/AskPage';
import { ReviewPage } from './pages/ReviewPage';
import { DecisionsPage } from './pages/DecisionsPage';
import { InspectorPage } from './pages/InspectorPage';
import { EvaluationPage } from './pages/EvaluationPage';
import { SettingsPage } from './pages/SettingsPage';

export default function App(): React.JSX.Element {
  return (
    <Routes>
      <Route element={<AppLayout />}>
        <Route index element={<OverviewPage />} />
        <Route path="ingest" element={<IngestPage />} />
        <Route path="memories" element={<MemoriesPage />} />
        <Route path="graph" element={<GraphPage />} />
        <Route path="timeline" element={<TimelinePage />} />
        <Route path="ask" element={<AskPage />} />
        <Route path="review" element={<ReviewPage />} />
        <Route path="decisions" element={<DecisionsPage />} />
        <Route path="inspector" element={<InspectorPage />} />
        <Route path="evaluation" element={<EvaluationPage />} />
        <Route path="settings" element={<SettingsPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}
