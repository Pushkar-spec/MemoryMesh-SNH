import React from 'react';
import { Loader2 } from 'lucide-react';

interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg';
  label?: string;
  className?: string;
}

export function LoadingSpinner({
  size = 'md',
  label,
  className = '',
}: LoadingSpinnerProps): React.JSX.Element {
  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-6 h-6',
    lg: 'w-8 h-8',
  };

  return (
    <div className={`flex items-center justify-center space-x-2 text-slate-400 ${className}`}>
      <Loader2 className={`${sizeClasses[size]} animate-spin text-indigo-500`} />
      {label && <span className="text-sm font-medium">{label}</span>}
    </div>
  );
}
