export interface Transaction {
    user_id: string;
    user_archetype: string;
    user_home_latitude: number;
    user_home_longitude: number;
    user_preferred_categories: string;
    transaction_id: string;
    timestamp: string;
    amount: number;
    merchant_id: string;
    merchant_name: string;
    merchant_category: string;
    merchant_latitude: number;
    merchant_longitude: number;
    transaction_type: string;
    transaction_latitude: number;
    transaction_longitude: number;
    distance_from_home: number;
    device_id: string;
    device_type: string;
    is_trusted_device: boolean;
    cvv_provided: boolean;
    three_d_secure: string;
    auth_attempts: number;
    is_fraud: boolean;
  }

  export interface PredictionResult {
    pred: string;
    confidence: number;
  }
  
  export interface Predictions {
    svm: PredictionResult;
    logisticRegression: PredictionResult;
    naiveBayes: PredictionResult;
    xgboost: PredictionResult;
    actualLabel: string;
  }

export interface ModelMetrics {
  correct: number;
  total: number;
  accuracy: number;
  truePositives: number;
  trueNegatives: number;
  falsePositives: number;
  falseNegatives: number;
  precision: number;
  recall: number;
  f1Score: number;
}

export interface StatsState {
  modelAccuracy: number;
  fraudCount: number;
  notFraudCount: number;
  totalPredictions: number;
  perModelStats: {
    svm: ModelMetrics;
    logisticRegression: ModelMetrics;
    naiveBayes: ModelMetrics;
    xgboost: ModelMetrics;
  };
}