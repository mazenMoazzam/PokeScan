import React from 'react';
import ImageUploader from './ImageUploader';
import './index.css';
import exampleImage from './realll.JPG';

function App() {
  return (
    <div className="App min-h-screen flex items-center justify-center bg-gradient-to-b from-blue-500 to-blue-700">
      <div className="bg-white shadow-2xl rounded-lg p-12 max-w-2xl w-full">
        <header className="mb-8">
          <h1 className="text-4xl font-bold text-blue-900 text-center">
            PokéScan
          </h1>
          <p className="text-gray-600 text-center mt-2">
            Upload an image to see if it's a real or fake Pokémon card.
          </p>
        </header>

        <ImageUploader />

        {/* Example Image Section */}
        <div className="mt-8 text-center">
          <h2 className="text-lg font-bold text-gray-700">Example:</h2>
          <img
            src={exampleImage}
            alt="Example Pokémon card"
            className="mt-4 mx-auto max-w-xs rounded-lg shadow-lg"
          />
          <p className="text-sm text-gray-500 mt-2">Example card image you can upload.</p>
        </div>
      </div>
    </div>
  );
}

export default App;
