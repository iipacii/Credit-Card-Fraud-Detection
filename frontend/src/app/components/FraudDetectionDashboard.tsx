'use client'

import { useState, useEffect, useRef } from 'react'
import TransactionDetails from './TransactionDetails'
import ModelPrediction from './ModelPrediction'
import Statistics from './Statistics'
import { Button } from './ui/button'
import { ArrowRight, RefreshCw } from 'lucide-react'
import ActualLabel from './ActualLabel'
import { Transaction, Predictions, StatsState, ModelMetrics } from '@/types'


export default function FraudDetectionDashboard() {
  const [transaction, setTransaction] = useState<Transaction | null>(null)
  const [predictions, setPredictions] = useState<Predictions | null>(null)
  const [loading, setLoading] = useState(false)
  const mountedRef = useRef(false)
  const fetchingRef = useRef(false)

  //add stats state to parent
  const [globalStats, setGlobalStats] = useState<StatsState>({
    modelAccuracy: 0,
    fraudCount: 0,
    notFraudCount: 0,
    totalPredictions: 0,
    perModelStats: {
      svm: createInitialMetrics(),
      logisticRegression: createInitialMetrics(),
      naiveBayes: createInitialMetrics(),
      xgboost: createInitialMetrics(),
    }
  });

 
// Update stats function
const updateStats = (predictions: Predictions) => {
  setGlobalStats(prevStats => {
    const newStats = { ...prevStats };
    const isFraud = predictions.actualLabel === 'Fraud';
    
    // Update global counts
    newStats.fraudCount += isFraud ? 1 : 0;
    newStats.notFraudCount += isFraud ? 0 : 1;
    newStats.totalPredictions += 1;

    // Update per-model metrics
    Object.entries(predictions).forEach(([model, prediction]) => {
      if (model === 'actualLabel') return;
      
      const modelStats = newStats.perModelStats[model as keyof typeof newStats.perModelStats];
      const isPredictedFraud = prediction.pred === 'Fraud';
      
      modelStats.total += 1;
      modelStats.correct += prediction.pred === predictions.actualLabel ? 1 : 0;
      
      if (isFraud && isPredictedFraud) modelStats.truePositives += 1;
      if (!isFraud && !isPredictedFraud) modelStats.trueNegatives += 1;
      if (!isFraud && isPredictedFraud) modelStats.falsePositives += 1;
      if (isFraud && !isPredictedFraud) modelStats.falseNegatives += 1;

      modelStats.accuracy = (modelStats.correct / modelStats.total) * 100;
      modelStats.precision = modelStats.truePositives / (modelStats.truePositives + modelStats.falsePositives);
      modelStats.recall = modelStats.truePositives / (modelStats.truePositives + modelStats.falseNegatives);
      modelStats.f1Score = 2 * (modelStats.precision * modelStats.recall) / (modelStats.precision + modelStats.recall);
    });

    return newStats;
  });
};

  const fetchTransaction = async () => {
    console.log('fetchTransaction called')
    if (fetchingRef.current || !mountedRef.current) return
    fetchingRef.current = true
    setLoading(true)

    try {
      const transResponse = await fetch('/api/transaction')
      const transactionData: Transaction = await transResponse.json()
      
      //convert is_fraud to boolean if it's a string
      const isFraud = typeof transactionData.is_fraud === 'string' 
        ? (transactionData.is_fraud as string).toLowerCase() === 'true'
        : Boolean(transactionData.is_fraud)
      
      console.log('Processed transaction data:', {
        ...transactionData,
        is_fraud: isFraud
      })
      
      if (!mountedRef.current) return
    

const [svm, lr, nb, xgb] = await Promise.all([
  fetch('/api/svm', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ ...transactionData, is_fraud: isFraud })
  }).then(async r => {
    console.log('SVM Raw Response:', r);
    const data = await r.json();
    console.log('SVM Parsed Response:', data);
    return data;
  }),
  fetch('/api/logisticregression', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ ...transactionData, is_fraud: isFraud })
  }).then(async r => {
    console.log('Logistic Regression Raw Response:', r);
    const data = await r.json();
    console.log('Logistic Regression Parsed Response:', data);
    return data;
  }),
  fetch('/api/naivebayes', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ ...transactionData, is_fraud: isFraud })
  }).then(async r => {
    console.log('Naive Bayes Raw Response:', r);
    const data = await r.json();
    console.log('Naive Bayes Parsed Response:', data);
    return data;
  }),
  fetch('/api/xgboost', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ ...transactionData, is_fraud: isFraud })
  }).then(async r => {
    console.log('XGBoost Raw Response:', r);
    const data = await r.json();
    const confidence = Math.max(...data.probabilities) * 100;
    console.log('XGBoost Parsed Response:', data);
    return {...data, confidence: confidence.toFixed(2)};
  })
])

      if (!mountedRef.current) return

      setTransaction({
        ...transactionData,
        is_fraud: isFraud
      })
      
      const actualLabel = isFraud ? 'Fraud' : 'Not Fraud'
      console.log('Setting actualLabel:', actualLabel)
      
      setPredictions({
        svm: {
          pred: svm.prediction,
          confidence: Math.max(...svm.probabilities) * 100
        },
        logisticRegression: {
          pred: lr.prediction,
          confidence: Math.max(...lr.probabilities) * 100
        },
        naiveBayes: {
          pred: nb.prediction,
          confidence: Math.max(...nb.probabilities) * 100
        },
        xgboost: {
          pred: xgb.prediction,
          confidence: Math.max(...xgb.probabilities) * 100
        },
        actualLabel
      });

      updateStats({
        svm: {
          pred: svm.prediction,
          confidence: Math.max(...svm.probabilities) * 100
        },
        logisticRegression: {
          pred: lr.prediction,
          confidence: Math.max(...lr.probabilities) * 100
        },
        naiveBayes: {
          pred: nb.prediction,
          confidence: Math.max(...nb.probabilities) * 100
        },
        xgboost: {
          pred: xgb.prediction,
          confidence: Math.max(...xgb.probabilities) * 100
        },
        actualLabel
      });

    } catch (error) {
      console.error('Fetch error:', error)
    } finally {
      if (mountedRef.current) {
        setLoading(false)
        fetchingRef.current = false
      }
    }
  }

  useEffect(() => {
    console.log('useEffect running')
    mountedRef.current = true
    fetchTransaction()
    return () => {
      mountedRef.current = false
    }
  }, [])

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD'
    }).format(amount)
  }

  const formatDateTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleString()
  }

  const getTransactionDetails = (transaction: Transaction | null) => {
    if (!transaction) return []

    return [
      { label: 'Transaction ID', value: transaction.transaction_id },
      { label: 'Amount', value: formatCurrency(transaction.amount) },
      { label: 'Timestamp', value: formatDateTime(transaction.timestamp) },
      { label: 'Merchant', value: transaction.merchant_name },
      { label: 'Category', value: transaction.merchant_category },
      { label: 'Type', value: transaction.transaction_type },
      { label: 'Distance from Home', value: `${transaction.distance_from_home} miles` },
      { label: 'Device Type', value: transaction.device_type },
      { label: 'Trusted Device', value: transaction.is_trusted_device ? 'Yes' : 'No' },
      { label: '3D Secure', value: transaction.three_d_secure },
      { label: 'Auth Attempts', value: transaction.auth_attempts.toString() }
    ]
  }

  console.log('Passing to ActualLabel:', transaction?.is_fraud)

  return (
    <div className="max-w-7xl mx-auto space-y-8 p-4">
      <div className="flex justify-between items-center mb-8">
        <h1 className="text-4xl font-bold">Credit Card Fraud Detection</h1>
        <Button 
          onClick={fetchTransaction} 
          className="bg-primary hover:bg-primary/90 text-primary-foreground font-bold py-2 px-4 rounded-full transition-all duration-200 ease-in-out transform hover:scale-105"
        >
          Analyze Next Transaction
          <ArrowRight className="ml-2 h-5 w-5" />
        </Button>
      </div>
      
      {loading ? (
        <div className="text-center">
          <RefreshCw className="animate-spin h-8 w-8 mx-auto" />
          <p className="mt-2">Loading transaction data...</p>
        </div>
      ) : (
        <>
<TransactionDetails details={transaction ? getTransactionDetails(transaction) : []} />
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
            <ModelPrediction modelName="SVM" prediction={predictions?.svm} />
            <ModelPrediction modelName="Logistic Regression" prediction={predictions?.logisticRegression} />
            <ModelPrediction modelName="Naive Bayes" prediction={predictions?.naiveBayes} />
            <ModelPrediction modelName="XGBoost" prediction={predictions?.xgboost} />
            <ActualLabel actualLabel={transaction?.is_fraud} />
          </div>
          <Statistics stats={globalStats} predictions={predictions} />
        </>
      )}
    </div>
  )
}

function createInitialMetrics(): ModelMetrics {
  return {
    total: 0,
    correct: 0,
    truePositives: 0,
    trueNegatives: 0,
    falsePositives: 0,
    falseNegatives: 0,
    accuracy: 0,
    precision: 0,
    recall: 0,
    f1Score: 0
  };
}

