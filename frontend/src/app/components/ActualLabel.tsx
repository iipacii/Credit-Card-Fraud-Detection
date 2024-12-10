import { Card, CardContent, CardHeader, CardTitle } from './ui/card'
import { Badge } from './ui/badge'
import { AlertTriangle, CheckCircle, Tag } from 'lucide-react'

interface ActualLabelProps {
  actualLabel?: boolean | string  //Accept boolean or string
}

export default function ActualLabel({ actualLabel }: ActualLabelProps) {
  console.log('ActualLabel received prop:', actualLabel)
  console.log('ActualLabel prop type:', typeof actualLabel)
  
  if (actualLabel === undefined) {
    return null
  }


  const isFraud = typeof actualLabel === 'string' 
    ? actualLabel.toLowerCase() !== 'false' 
    : actualLabel
    
  console.log('Final isFraud value:', isFraud)
  const displayText = isFraud ? 'Fraud' : 'Not Fraud'
  console.log('Display text:', displayText)

  return (
    <Card className="model-card">
      <CardHeader>
        <CardTitle className="flex items-center justify-center">
          <Tag className="mr-2" />
          Actual Label
        </CardTitle>
      </CardHeader>
      <CardContent className="text-center">
        <Badge 
          className={`text-lg py-2 px-4 ${
            isFraud ? 'bg-red-500 text-white ' : 'bg-green-500 text-white'
          }`}
        >
          {isFraud ? (
            <AlertTriangle className="inline-block mr-2" />
          ) : (
            <CheckCircle className="inline-block mr-2" />
          )}
          {displayText}
        </Badge>
      </CardContent>
    </Card>
  )
}

