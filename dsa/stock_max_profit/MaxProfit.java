/*
Given an array, find the max sum of subarray.
Input: [2,-3,4,-1,2,3,-7,-2]
Output: 8
Explanation: [4,-1,2,3] = 8

array - subarray - sum -> prefix sum 
*/


public class MaxProfit {
    
    static int maxProfit(int[] prices) {
        // special cases
        if (prices == null || prices.length == 0) {
            return 0;
        }

        int previousMinPrice = Integer.MAX_VALUE;
        int maxProfit = 0; 

        for (int currPrice: prices) {
            previousMinPrice = currPrice < previousMinPrice ? currPrice : previousMinPrice;
            
            maxProfit = (currPrice - previousMinPrice) > maxProfit ? currPrice - previousMinPrice : maxProfit;

            System.out.printf("%d, %d\n", previousMinPrice, maxProfit);
        }
        return maxProfit;
    }

    public static void main(String[] args) {
        int[] prices = {102,99,103,102,104,107,100,98};
        int res = MaxProfit.maxProfit(prices);
        System.out.println(res);

    }
}
