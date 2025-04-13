/*
Given an array, find the max sum of subarray.
Input: [2,-3,4,-1,2,3,-7,-2]
Output: 8
Explanation: [4,-1,2,3] = 8

array - subarray - sum -> prefix sum 
*/


public class MaxSubarray {
    
    static int maxSubArray(int[] nums) {
        // special case
        if (nums == null || nums.length == 0) {
            return 0;
        }

        int max = Integer.MIN_VALUE;  // result 
        int prefixSum = 0;  //largest sum so far
        int minPrefixSum = 0;   // smallest sum so far

        for (int i = 0; i < nums.length; i++) {
            prefixSum += nums[i];
            max = Math.max(max, prefixSum - minPrefixSum);
            
            System.out.printf("%d, %d, %d\n", prefixSum, nums[i], max);
            minPrefixSum = Math.min(minPrefixSum, prefixSum);
        }
        return max;
    }

    public static void main(String[] args) {
        int[] arr = {2,-3,4,-1,2,3,-7,-2};
        int res = MaxSubarray.maxSubArray(arr);
        System.out.println(res);

    }
}
