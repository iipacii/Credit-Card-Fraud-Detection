'use client'

import { Card, CardContent, CardHeader, CardTitle } from './ui/card'
// import { Progress } from './ui/progress'
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip, BarChart, Bar, XAxis, YAxis, CartesianGrid } from 'recharts'
import {  PieChartIcon, BarChart2 } from 'lucide-react'
import { Predictions } from '@/types'

interface ModelStats {
  correct: number;
  total: number;
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
}

interface Stats {
  modelAccuracy: number;
  fraudCount: number;
  notFraudCount: number;
  totalPredictions: number;
  perModelStats: {
    svm: ModelStats;
    logisticRegression: ModelStats;
    naiveBayes: ModelStats;
    xgboost: ModelStats;
  };
}

interface StatisticsProps {
  stats: Stats;
  predictions: Predictions | null;
}


export default function Statistics({ stats }: StatisticsProps) {
  const COLORS = ['#FF4444', '#4ECDC4'];

  const pieChartData = [
    { name: 'Fraud', value: stats.fraudCount },
    { name: 'Not Fraud', value: stats.notFraudCount },
  ];

  const barChartData = Object.entries(stats.perModelStats).map(([name, metrics]) => ({
    name: name === 'logisticRegression' ? 'Log. Reg.' : name.toUpperCase(),
    accuracy: metrics.accuracy,
    precision: metrics.precision * 100,
    recall: metrics.recall * 100,
    f1Score: metrics.f1Score * 100,
  }));

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center">
            <PieChartIcon className="mr-2" />
            Fraud vs Not Fraud Distribution
          </CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={pieChartData}
                cx="50%"
                cy="50%"
                labelLine={false}
                outerRadius={100}
                fill="#8884d8"
                dataKey="value"
              >
                {pieChartData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center">
            <BarChart2 className="mr-2" />
            Model Metrics Comparison
          </CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart 
              data={barChartData}
              layout="vertical"
              margin={{ top: 5, right: 30, left: 60, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" domain={[0, 100]} />
              <YAxis type="category" dataKey="name" width={80} />
              <Tooltip />
              <Legend verticalAlign="top" height={36} />
              <Bar dataKey="accuracy" fill="#8884d8" name="Accuracy" />
              <Bar dataKey="precision" fill="#82ca9d" name="Precision" />
              <Bar dataKey="recall" fill="#ffc658" name="Recall" />
              <Bar dataKey="f1Score" fill="#ff7300" name="F1 Score" />
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </div>
  );
}

