import { Card, CardContent, CardHeader, CardTitle } from './ui/card'
import { Badge } from './ui/badge'
import { AlertTriangle, CheckCircle, BarChart } from 'lucide-react'
import { PredictionResult } from '@/types'

interface ModelPredictionProps {
  modelName: string
  prediction: PredictionResult | undefined
}

export default function ModelPrediction({ modelName, prediction }: ModelPredictionProps) {
  const isFraud = prediction?.pred === 'Fraud'
  const confidenceScore = (prediction?.confidence ?? 0)

  return (
    <Card className="model-card">
      <CardHeader>
        <CardTitle className="flex items-center justify-center">
          <BarChart className="mr-2" />
          {modelName}
        </CardTitle>
      </CardHeader>
      <CardContent className="text-center space-y-4">
        <Badge 
          className={`text-lg py-2 px-4 ${
            isFraud ? 'bg-red-500 text-white' : 'bg-green-500 text-white'
          }`}
        >
          {isFraud ? (
            <AlertTriangle className="inline-block mr-2" />
          ) : (
            <CheckCircle className="inline-block mr-2" />
          )}
          {prediction?.pred}
        </Badge>
        <div>
          <p className="text-sm text-muted-foreground">Confidence Score</p>
          <p className="text-xl font-bold">{confidenceScore.toFixed(2)}%</p>
        </div>
      </CardContent>
    </Card>
  )
}

