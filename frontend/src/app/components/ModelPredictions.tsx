import { Card, CardContent, CardHeader, CardTitle } from './ui/card'
import { Badge } from './ui/badge'

interface ModelPredictionsProps {
  predictions: {
    model1: string
    model2: string
    model3: string
    model4: string
    actualLabel: string
  } | null
}

export default function ModelPredictions({ predictions }: ModelPredictionsProps) {
  if (!predictions) return null

  const renderPrediction = (label: string, prediction: string) => (
    <div className="flex items-center justify-between">
      <span>{label}:</span>
      <Badge className={prediction === 'Fraud' ? 'bg-red-500' : 'bg-blue-500'}>
        {prediction}
      </Badge>
    </div>
  )

  return (
    <Card className="bg-gray-800 text-white">
      <CardHeader>
        <CardTitle>Model Predictions</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {renderPrediction('Model 1', predictions.model1)}
        {renderPrediction('Model 2', predictions.model2)}
        {renderPrediction('Model 3', predictions.model3)}
        {renderPrediction('Model 4', predictions.model4)}
        {renderPrediction('Actual Label', predictions.actualLabel)}
      </CardContent>
    </Card>
  )
}

