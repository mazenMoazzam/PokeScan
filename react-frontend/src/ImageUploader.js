import React, { useState } from 'react';
import axios from 'axios';

function ImageUploader() {
    const [file, setFile] = useState(null);
    const [prediction, setPrediction] = useState('');
    const [loading, setLoading] = useState(false);

    const handleFileChange = (event) => {
        setFile(event.target.files[0]);
    };

    const handleSubmit = async (event) => {
        event.preventDefault();

        if (!file) {
            alert("Please upload an image first.");
            return;
        }

        setLoading(true);

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await axios.post('http://127.0.0.1:5000/predict', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data'
                }
            });

            setPrediction(response.data.Prediction === 1 ? 'Real' : 'Fake');
        } catch (error) {
            console.error('Error uploading file:', error);
            setPrediction('Error occurred');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="p-4 max-w-md mx-auto">
            <h1 className="text-xl font-bold mb-4">Pokémon Card Detector</h1>
            <form onSubmit={handleSubmit} className="space-y-4">
                <input
                    type="file"
                    onChange={handleFileChange}
                    className="border p-2 w-full"
                />
                <button
                    type="submit"
                    disabled={loading}
                    className="bg-blue-500 text-white p-2 rounded hover:bg-blue-600"
                >
                    {loading ? 'Uploading...' : 'Submit'}
                </button>
            </form>
            {prediction && <h2 className="mt-4 text-lg font-semibold">Prediction: {prediction}</h2>}
        </div>
    );
}

export default ImageUploader;