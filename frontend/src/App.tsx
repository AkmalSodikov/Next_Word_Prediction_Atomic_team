import { useState, useRef, useEffect } from 'react';
import { Input } from '@/components/ui/input';
import { Alert, AlertDescription } from '@/components/ui/alert';

const API_URL = 'http://localhost:8000';

function App() {
  const [value, setValue] = useState('');
  const [prediction, setPrediction] = useState('');
  const [suggestions, setSuggestions] = useState([]);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const inputRef = useRef(null);

  useEffect(() => {
    const fetchPredictions = async () => {
      if (!value.trim()) {
        setPrediction('');
        setSuggestions([]);
        setError('');
        return;
      }

      setLoading(true);
      setError('');

      try {
        const response = await fetch(`${API_URL}/predict?text=${value.trim()}&k=5`);

        if (!response.ok) {
          throw new Error('Failed to fetch predictions');
        }

        const data = await response.json();

        if (data.predictions && data.predictions.length > 0) {
          const words = data.predictions.map((p) => p.word);
          setSuggestions(words);
          setSelectedIndex(0);
          setPrediction(words[0]);
        } else {
          setPrediction('');
          setSuggestions([]);
        }
      } catch (err) {
        setError(err.message);
        setPrediction('');
        setSuggestions([]);
      } finally {
        setLoading(false);
      }
    };

    const debounceTimer = setTimeout(fetchPredictions, 150);
    return () => clearTimeout(debounceTimer);
  }, [value]);

  const completePrediction = (word) => {
    const newValue = value + ' ' + word;
    setValue(newValue);
    setPrediction('');
    setSuggestions([]);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Tab' && prediction) {
      e.preventDefault();
      completePrediction(prediction);
    } else if (e.key === 'ArrowDown') {
      e.preventDefault();
      setSelectedIndex((prev) =>
        prev < suggestions.length - 1 ? prev + 1 : prev
      );
      if (suggestions.length > 0) {
        setPrediction(
          suggestions[Math.min(selectedIndex + 1, suggestions.length - 1)]
        );
      }
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setSelectedIndex((prev) => (prev > 0 ? prev - 1 : prev));
      if (suggestions.length > 0) {
        setPrediction(suggestions[Math.max(selectedIndex - 1, 0)]);
      }
    } else if (e.key === 'Enter' && suggestions.length > 0) {
      e.preventDefault();
      completePrediction(suggestions[selectedIndex]);
    }
  };

  const handleSuggestionClick = (word) => {
    completePrediction(word);
    inputRef.current?.focus();
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-gradient-to-br from-slate-50 to-slate-100 p-4">
      <div className="w-full max-w-2xl space-y-4">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-slate-800 mb-2">
            Atomic word
          </h1>
          <p className="text-slate-600">
            Next word predictor you have been waiting for
          </p>
        </div>

        <div className="relative">
          {prediction && !loading && (
            <div className="pointer-events-none absolute inset-0 flex items-center px-3">
              <span className="invisible">{value}</span>
              <span className="text-slate-400 ml-1">{prediction}</span>
            </div>
          )}

          <Input
            ref={inputRef}
            type="text"
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Start typing a sentence..."
            className="relative z-10 bg-white text-lg h-14 shadow-lg border-slate-200 focus-visible:ring-2 focus-visible:ring-blue-400"
          />

          {loading && (
            <div className="absolute right-3 top-1/2 -translate-y-1/2 z-20">
              <div className="w-5 h-5 border-2 border-slate-300 border-t-blue-500 rounded-full animate-spin" />
            </div>
          )}
        </div>

        {error && (
          <Alert variant="destructive">
            <AlertDescription>
              Error connecting to API: {error}. Make sure the backend is running
              on {API_URL}
            </AlertDescription>
          </Alert>
        )}

        {suggestions.length > 0 && !error && (
          <div className="rounded-lg border border-slate-200 bg-white shadow-lg overflow-hidden">
            {suggestions.map((word, idx) => (
              <button
                key={`${word}-${idx}`}
                onClick={() => handleSuggestionClick(word)}
                className={`w-full px-4 py-3 text-left transition-colors flex items-center justify-between ${
                  idx === selectedIndex
                    ? 'bg-blue-50 text-blue-900 border-l-4 border-blue-500'
                    : 'text-slate-700 hover:bg-slate-50'
                }`}
              >
                <span className="font-medium text-lg">{word}</span>
                {idx === selectedIndex && (
                  <kbd className="px-2 py-1 text-xs bg-blue-100 text-blue-700 rounded">
                    Tab
                  </kbd>
                )}
              </button>
            ))}
          </div>
        )}

        <div className="bg-white rounded-lg shadow p-4 border border-slate-200">
          <p className="text-sm text-slate-600 mb-2">
            <strong>How to use:</strong>
          </p>
          <ul className="text-sm text-slate-600 space-y-1">
            <li>• Type to see AI-powered word predictions</li>
            <li>
              • Press{' '}
              <kbd className="px-1.5 py-0.5 rounded bg-slate-200 text-slate-700 font-mono text-xs">
                Tab
              </kbd>{' '}
              to accept the top prediction
            </li>
            <li>
              • Use{' '}
              <kbd className="px-1.5 py-0.5 rounded bg-slate-200 text-slate-700 font-mono text-xs">
                ↑
              </kbd>{' '}
              <kbd className="px-1.5 py-0.5 rounded bg-slate-200 text-slate-700 font-mono text-xs">
                ↓
              </kbd>{' '}
              to navigate suggestions
            </li>
            <li>• Click any suggestion to select it</li>
          </ul>
        </div>
      </div>
    </div>
  );
}

export default App;
