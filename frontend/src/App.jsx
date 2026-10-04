import { Navigate, Route, Routes } from "react-router-dom";
import FaceToSketchPage from "./pages/FaceToSketchPage.jsx";
import HardRoutedPage from "./pages/HardRoutedPage.jsx";
import Landing from "./pages/Landing.jsx";
import { SystemPage } from "./pages/OtherPages.jsx";
import SoftMoEPage from "./pages/SoftMoEPage.jsx";
import UniversalPage from "./pages/UniversalPage.jsx";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Landing />} />
      <Route path="/universal" element={<UniversalPage />} />
      <Route path="/hard-routed" element={<HardRoutedPage />} />
      <Route path="/soft-moe" element={<SoftMoEPage />} />
      <Route path="/face-to-sketch" element={<FaceToSketchPage />} />
      <Route path="/system" element={<SystemPage />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
