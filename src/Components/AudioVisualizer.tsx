import React, { useEffect, useRef } from 'react';
import { AudioVisualizerProps } from '../types';

export const AudioVisualizer: React.FC<AudioVisualizerProps> = ({ isPlaying, volume }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const requestRef = useRef<number>(0);
  const angleRef = useRef<number>(0);

  const animate = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    const centerX = width / 2;
    const centerY = height / 2;
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Base radius
    const baseRadius = 60;
    // Dynamic radius based on volume (simulated visualizer)
    const dynamicRadius = baseRadius + (volume * 80);

    angleRef.current += 0.02;

    // Draw generic breathing circle if idle
    if (!isPlaying) {
       ctx.beginPath();
       ctx.arc(centerX, centerY, baseRadius + Math.sin(angleRef.current) * 5, 0, 2 * Math.PI);
       ctx.strokeStyle = 'rgba(148, 163, 184, 0.5)';
       ctx.lineWidth = 2;
       ctx.stroke();
       requestRef.current = requestAnimationFrame(animate);
       return;
    }

    // Draw active wavy circle
    ctx.beginPath();
    const spikes = 20; // Number of spikes
    const outerRadius = dynamicRadius;
    const innerRadius = baseRadius;
    
    for (let i = 0; i < spikes * 2; i++) {
       const r = (i % 2 === 0) ? outerRadius : innerRadius;
       // Add some rotation
       const currAngle = (Math.PI / spikes) * i + angleRef.current;
       const x = centerX + Math.cos(currAngle) * r;
       const y = centerY + Math.sin(currAngle) * r;
       if (i === 0) ctx.moveTo(x, y);
       else ctx.lineTo(x, y);
    }
    
    ctx.closePath();
    
    // Gradient fill
    const gradient = ctx.createRadialGradient(centerX, centerY, innerRadius * 0.5, centerX, centerY, outerRadius);
    gradient.addColorStop(0, 'rgba(168, 85, 247, 0.8)'); // Purple
    gradient.addColorStop(1, 'rgba(236, 72, 153, 0.4)'); // Pink

    ctx.fillStyle = gradient;
    ctx.fill();
    ctx.strokeStyle = 'rgba(236, 72, 153, 0.8)';
    ctx.lineWidth = 2;
    ctx.stroke();

    requestRef.current = requestAnimationFrame(animate);
  };

  useEffect(() => {
    requestRef.current = requestAnimationFrame(animate);
    return () => {
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
    };
  }, [isPlaying, volume]);

  return (
    <canvas 
      ref={canvasRef} 
      width={400} 
      height={400} 
      className="w-64 h-64 md:w-96 md:h-96"
    />
  );
};