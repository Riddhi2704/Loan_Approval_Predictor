import { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

const initialFormData = {
  no_of_dependents: 0,
  education: 1, // 1 for Graduate, 0 for Not Graduate
  self_employed: 0, // 0 for No, 1 for Yes
  income_annum: 500000,
  loan_amount: 1500000,
  loan_term: 5,
  cibil_score: 700,
  residential_assets_value: 1000000,
  commercial_assets_value: 0,
  luxury_assets_value: 200000,
  bank_asset_value: 300000,
};

function App() {
  const [formData, setFormData] = useState(initialFormData);

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [barWidth, setBarWidth] = useState(0);

  useEffect(() => {
    if (result) {
      setTimeout(() => setBarWidth(result.probability * 100), 100);
    } else {
      setBarWidth(0);
    }
  }, [result]);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: parseFloat(value),
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setResult(null);

    try {
      // Simulate network delay for animation effect
      await new Promise(resolve => setTimeout(resolve, 1500));
      const response = await axios.post('http://localhost:8000/predict', formData);
      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'An error occurred while connecting to the server.');
    } finally {
      setLoading(false);
    }
  };

  const handleRestart = () => {
    setResult(null);
    setError('');
    setFormData(initialFormData);
    setBarWidth(0);
  };

  return (
    <div className="app-wrapper">
      <div className="brand-header">
        <div className="brand-icon">💸</div>
        <h1 style={{ color: 'var(--primary-blue)', margin: 0, fontSize: '1.5rem', fontWeight: 800 }}>Predictify</h1>
      </div>

      <div className="app-container">
        <div className="header">
          <h1>Loan Approval Check</h1>
          <p>Instantly predict your loan eligibility using our advanced AI model.</p>
        </div>

        {!result && !error && (
          <form onSubmit={handleSubmit}>
            <div className="form-grid">
              <div className="form-group">
                <label>Number of Dependents</label>
                <input type="number" className="form-control" name="no_of_dependents" value={formData.no_of_dependents} onChange={handleChange} required min="0" />
              </div>
              
              <div className="form-group">
                <label>Education</label>
                <select className="form-control" name="education" value={formData.education} onChange={handleChange}>
                  <option value={1}>Graduate</option>
                  <option value={0}>Not Graduate</option>
                </select>
              </div>

              <div className="form-group">
                <label>Self Employed</label>
                <select className="form-control" name="self_employed" value={formData.self_employed} onChange={handleChange}>
                  <option value={1}>Yes</option>
                  <option value={0}>No</option>
                </select>
              </div>

              <div className="form-group">
                <label>Annual Income (₹)</label>
                <input type="number" className="form-control" name="income_annum" value={formData.income_annum} onChange={handleChange} required min="0" />
              </div>

              <div className="form-group">
                <label>Loan Amount (₹)</label>
                <input type="number" className="form-control" name="loan_amount" value={formData.loan_amount} onChange={handleChange} required min="0" />
              </div>

              <div className="form-group">
                <label>Loan Term (Years)</label>
                <input type="number" className="form-control" name="loan_term" value={formData.loan_term} onChange={handleChange} required min="1" max="30" />
              </div>

              <div className="form-group">
                <label>CIBIL Score</label>
                <input type="number" className="form-control" name="cibil_score" value={formData.cibil_score} onChange={handleChange} required min="300" max="900" />
              </div>

              <div className="form-group">
                <label>Residential Assets (₹)</label>
                <input type="number" className="form-control" name="residential_assets_value" value={formData.residential_assets_value} onChange={handleChange} required min="0" />
              </div>
              
              <div className="form-group">
                <label>Commercial Assets (₹)</label>
                <input type="number" className="form-control" name="commercial_assets_value" value={formData.commercial_assets_value} onChange={handleChange} required min="0" />
              </div>

              <div className="form-group">
                <label>Luxury Assets (₹)</label>
                <input type="number" className="form-control" name="luxury_assets_value" value={formData.luxury_assets_value} onChange={handleChange} required min="0" />
              </div>

              <div className="form-group">
                <label>Bank Assets (₹)</label>
                <input type="number" className="form-control" name="bank_asset_value" value={formData.bank_asset_value} onChange={handleChange} required min="0" />
              </div>
            </div>

            <button type="submit" className="btn-submit" disabled={loading}>
              Get Prediction
            </button>
          </form>
        )}

        {error && (
          <div style={{ textAlign: 'center' }}>
            <div style={{ color: 'red', marginTop: '20px', fontWeight: 'bold' }}>{error}</div>
            <button className="btn-restart" onClick={handleRestart}>Try Again</button>
          </div>
        )}

        {result && (
          <div className={`result-container ${result.prediction === 'Approved' ? 'approved' : 'rejected'}`}>
            <div className="stamp">
              {result.prediction}
            </div>
            <p className="result-text">
              {result.prediction === 'Approved' 
                ? 'Congratulations! Based on our AI analysis, your loan application looks solid and is likely to be approved.' 
                : 'Based on our AI analysis, your current profile might lead to a loan rejection. Consider improving your CIBIL score or adjusting the loan amount.'}
            </p>
            
            <div className="confidence-bar-container">
              <div className="confidence-bar" style={{ width: `${barWidth}%` }}></div>
            </div>
            <div className={`confidence-text ${result.prediction === 'Approved' ? 'approved-text' : 'rejected-text'}`} style={{ color: result.prediction === 'Approved' ? 'var(--primary-green)' : 'var(--primary-red)' }}>
              AI Model Confidence: {(result.probability * 100).toFixed(1)}%
            </div>
            
            <button className="btn-restart" onClick={handleRestart}>Check Another Loan</button>
          </div>
        )}
      </div>

      {loading && (
        <div className="loading-overlay">
          <div className="scanning-box">
            <div className="scanner-line"></div>
          </div>
          <div className="loading-text">Analyzing Application...</div>
        </div>
      )}
    </div>
  );
}

export default App;
