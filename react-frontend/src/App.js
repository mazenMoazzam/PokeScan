import React from 'react';
import ImageUploader from './ImageUploader';  // Import the ImageUploader component
import './index.css';  // Ensure Tailwind CSS is imported

function App() {
  return (
    <div className="App">
      <header className="p-4 max-w-md mx-auto">
        <h1 className="text-xl font-bold mb-4">Fake Pokémon Card Detector</h1>
        <ImageUploader />
      </header>
    </div>
  );
}

export default App;

