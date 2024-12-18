import React, { useState, useEffect } from "react";
import "./App.css";
import Papa from "papaparse";

function App() {
  const [reviews, setReviews] = useState([]);
  const [productUrl, setProductUrl] = useState("");
  const [error, setError] = useState("");
  const [loadingScrape, setLoadingScrape] = useState(false);
  const [loadingPredict, setLoadingPredict] = useState(false);

  // Function to load reviews
  const loadReviews = async (csvFile) => {
    try {
      const response = await fetch(`http://127.0.0.1:5000/get_csv?file=${csvFile}`);
      const data = await response.text();
  
      // Parse the CSV using PapaParse
      const parsedData = Papa.parse(data, {
        header: true,
        skipEmptyLines: true,
      });
  
      const reviews = parsedData.data.map((row) => ({
        title: row["Review Title"]?.trim(),
        text: row["Review Text"]?.trim(),
        rating: row["Star Rating"]?.trim(),
        verified: row["Verified Purchase"]?.trim(),
        prediction: row["Prediction"]?.trim() || "N/A",
      }));
  
      setReviews(reviews);
    } catch (err) {
      console.error("Fetch error:", err);
      setError(`Error loading reviews: ${err.message}`);
    }
  };

  // Function to scrape reviews
  const scrapeReviews = async () => {
    if (!productUrl.trim()) {
      setError("Please enter a valid product URL.");
      return;
    }

    try {
      setError("");
      setLoadingScrape(true);

      const response = await fetch("http://127.0.0.1:5000/scrape_reviews", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: productUrl }),
      });

      const result = await response.json();
      console.log("Scraping Result:", result);

      if (response.ok) {
        await loadReviews("reviews.csv"); // Load the scraped reviews
      } else {
        throw new Error(result.error || "Scraping failed.");
      }
    } catch (err) {
      console.error("Scraping error:", err);
      setError(`Error scraping reviews: ${err.message}`);
    } finally {
      setLoadingScrape(false);
    }
  };

  // Function to predict reviews
  const predictReviews = async () => {
    try {
      setError("");
      setLoadingPredict(true);

      const response = await fetch("http://127.0.0.1:5000/predict_reviews", {
        method: "POST",
      });

      const result = await response.json();
      console.log("Prediction Result:", result);

      if (response.ok) {
        await loadReviews("reviews_with_predictions.csv"); // Load the predicted reviews
      } else {
        throw new Error(result.error || "Prediction failed.");
      }
    } catch (err) {
      console.error("Prediction error:", err);
      setError(`Error predicting reviews: ${err.message}`);
    } finally {
      setLoadingPredict(false);
    }
  };

  useEffect(() => {
    loadReviews("reviews.csv"); // Load reviews when the component mounts
  }, []);

  return (
    <div className="App">
      <h1>Amazon Reviews Scraper and Predictor</h1>

      {/* Error Message */}
      {error && <p style={{ color: "red" }}>{error}</p>}

      {/* Input for Product URL */}
      <div>
        <input
          type="text"
          placeholder="Enter Amazon Product URL"
          value={productUrl}
          onChange={(e) => setProductUrl(e.target.value)}
          style={{ width: "400px", padding: "8px", marginRight: "10px" }}
        />
        <button onClick={scrapeReviews} disabled={loadingScrape}>
          {loadingScrape ? "Scraping..." : "Scrape Reviews"}
        </button>
      </div>

      {/* Predict Button */}
      <div style={{ marginTop: "10px" }}>
        <button onClick={predictReviews} disabled={loadingPredict}>
          {loadingPredict ? "Predicting..." : "Predict Real or Fake"}
        </button>
      </div>

      {/* Reviews Cards */}
      <div style={{ marginTop: "20px" }}>
        {reviews.length > 0 ? (
          reviews.map((review, index) => (
            <div
              key={index}
              style={{
                border: "1px solid #ccc",
                padding: "10px",
                margin: "10px 0",
                borderRadius: "5px",
                boxShadow: "0 2px 4px rgba(0, 0, 0, 0.1)",
              }}
            >
              <h3>{review.title}</h3>
              <p><strong>Review:</strong> {review.text}</p>
              <p><strong>Rating:</strong> {review.rating}</p>
              <p>
                <strong>Verified Purchase:</strong>{" "}
                {review.verified === "True" ? "Yes" : "No"}
              </p>
              <p>
                <strong>Prediction:</strong>{" "}
                {review.prediction === "real"
                  ? <span style={{ color: "green" }}>Real</span>
                  : review.prediction === "fake"
                  ? <span style={{ color: "red" }}>Fake</span>
                  : "N/A"}
              </p>
            </div>
          ))
        ) : (
          <p>No reviews available. Please scrape reviews.</p>
        )}
      </div>
    </div>
  );
}

export default App;
