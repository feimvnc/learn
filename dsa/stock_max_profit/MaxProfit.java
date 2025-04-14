/*
Given an array, find the max difference.
Input: [102,99,103,102,104,107,100,98]
Output: 8
Explanation: [99,107] = 8

minimun_point - max_point -> max profit
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
