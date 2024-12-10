import { CreditCard, Smartphone, Shield, Store } from 'lucide-react'
import { Badge } from './ui/badge'

interface TransactionDetailsProps {
  details?: Array<{ label: string; value: string }>;  // Make optional
}

export default function TransactionDetails({ details = [] }: TransactionDetailsProps) {  //add default empty array
  //early return if no details
  if (!details?.length) {
    return (
      <div className="text-center p-4">
        <p className="text-gray-500">No transaction details available</p>
      </div>
    );
  }

  // Group details into categories
  const transactionInfo = details.slice(0, 4);
  const merchantInfo = details.slice(4, 7);
  const deviceInfo = details.slice(7, 9);
  const securityInfo = details.slice(9);

  return (
    <div className="jumbotron mb-8">
      <h2 className="text-2xl font-bold mb-4">Transaction Details</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <div className="space-y-4">
          <div>
            <h3 className="text-lg font-semibold mb-2 flex items-center">
              <CreditCard className="mr-2" /> Transaction Info
            </h3>
            {transactionInfo.map((item, index) => (
              <p key={index}><span className="font-medium">{item.label}:</span> {item.value}</p>
            ))}
          </div>
          <div>
            <h3 className="text-lg font-semibold mb-2 flex items-center">
              <Store className="mr-2" /> Merchant Details
            </h3>
            {merchantInfo.map((item, index) => (
              <p key={index}><span className="font-medium">{item.label}:</span> {item.value}</p>
            ))}
          </div>
        </div>
        <div className="space-y-4">
          <div>
            <h3 className="text-lg font-semibold mb-2 flex items-center">
              <Smartphone className="mr-2" /> Device Info
            </h3>
            {deviceInfo.map((item, index) => (
              <p key={index}>
                <span className="font-medium">{item.label}:</span>
                {item.label.includes('Trusted') ? (
                  <Badge className={`ml-2 ${item.value === 'Yes' ? 'bg-green-500' : 'bg-red-500'}`}>
                    {item.value}
                  </Badge>
                ) : (
                  ` ${item.value}`
                )}
              </p>
            ))}
          </div>
          <div>
            <h3 className="text-lg font-semibold mb-2 flex items-center">
              <Shield className="mr-2" /> Security Info
            </h3>
            {securityInfo.map((item, index) => (
              <p key={index}><span className="font-medium">{item.label}:</span> {item.value}</p>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

