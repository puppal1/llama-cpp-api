/**
 * TypeScript implementation of shared load rules between UI and backend
 */

export interface SystemLimits {
  maxContextSize: number;
  minContextSize: number;
  maxCpuThreads: number;
  memorySafetyMargin: number;
}

export const DEFAULT_SYSTEM_LIMITS: SystemLimits = {
  maxContextSize: 4096,
  minContextSize: 512,
  maxCpuThreads: 16,
  memorySafetyMargin: 0.2
};

export interface MemoryBreakdown {
  baseModelMb: number;
  kvCacheMb: number;
  computeBufferMb: number;
  outputBufferMb: number;
  safetyBufferMb: number;
}

export interface MemoryRequirements {
  totalMb: number;
  breakdown: MemoryBreakdown;
}

export class MemoryCalculator {
  static calculateKvCacheSize(contextSize: number, nLayers: number): number {
    const bytesPerToken = 2; // float16
    const kvCacheBytes = 2 * contextSize * bytesPerToken * nLayers;
    return kvCacheBytes / (1024 * 1024); // Convert to MB
  }

  static calculateTotalMemoryRequired(
    modelSizeMb: number,
    contextSize: number,
    nLayers: number,
    safetyMargin: number = DEFAULT_SYSTEM_LIMITS.memorySafetyMargin
  ): MemoryRequirements {
    // Base model memory
    const baseMemory = modelSizeMb;
    
    // KV cache
    const kvCache = this.calculateKvCacheSize(contextSize, nLayers);
    
    // Compute buffers (empirical measurements)
    const computeBuffer = 164.01;
    const outputBuffer = 0.12;
    
    // Calculate total with safety margin
    const subtotal = baseMemory + kvCache + computeBuffer + outputBuffer;
    const safetyBuffer = subtotal * safetyMargin;
    const total = subtotal + safetyBuffer;
    
    return {
      totalMb: total,
      breakdown: {
        baseModelMb: baseMemory,
        kvCacheMb: kvCache,
        computeBufferMb: computeBuffer,
        outputBufferMb: outputBuffer,
        safetyBufferMb: safetyBuffer
      }
    };
  }
}

export class LoadRules {
  static getValidContextSizes(
    minSize: number = DEFAULT_SYSTEM_LIMITS.minContextSize,
    maxSize: number = 32768
  ): number[] {
    const sizes: number[] = [];
    let size = minSize;
    while (size <= maxSize) {
      sizes.push(size);
      size *= 2;
    }
    return sizes;
  }

  static getRecommendedThreadCount(): number {
    const cpuCount = navigator.hardwareConcurrency || 4;
    // Use power of 2 up to 16 threads
    return Math.min(16, Math.pow(2, Math.floor(Math.log2(cpuCount))));
  }

  static validateContextSize(
    requestedSize: number,
    modelMaxContext: number,
    systemMaxContext: number = DEFAULT_SYSTEM_LIMITS.maxContextSize
  ): [number, string | null] {
    const validSizes = this.getValidContextSizes();
    
    // Find nearest valid size
    const nearestSize = validSizes.reduce((prev, curr) => 
      Math.abs(curr - requestedSize) < Math.abs(prev - requestedSize) ? curr : prev
    );
    
    // Apply limits
    const finalSize = Math.min(nearestSize, modelMaxContext, systemMaxContext);
    
    // Generate warning if adjusted
    let warning = null;
    if (finalSize !== requestedSize) {
      warning = `Context size adjusted from ${requestedSize} to ${finalSize} to match valid size and limits`;
    }
    
    return [finalSize, warning];
  }

  static canLoadModel(
    modelSizeMb: number,
    contextSize: number,
    nLayers: number,
    availableMemoryMb: number,
    safetyMargin: number = DEFAULT_SYSTEM_LIMITS.memorySafetyMargin
  ): [boolean, string] {
    const requirements = MemoryCalculator.calculateTotalMemoryRequired(
      modelSizeMb,
      contextSize,
      nLayers,
      safetyMargin
    );
    
    if (requirements.totalMb > availableMemoryMb) {
      return [
        false,
        `Insufficient memory. Required: ${requirements.totalMb.toFixed(1)}MB, Available: ${availableMemoryMb.toFixed(1)}MB`
      ];
    }
    
    return [true, "Model can be loaded with specified parameters"];
  }
} 