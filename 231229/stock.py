class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        min_so_far = None 
        max_profit = None 


        for i, price in enumerate(prices): 
            if min_so_far is None: 
                min_so_far = price 
            else: 
                min_so_far = min(price, min_so_far)
            
            if max_profit is None: max_profit = 0
            else: 
                # max profit that can be achieved at step i is by buying 
                # at price min_so_far and selling at price.
                max_profit = max(max_profit, price-min_so_far)

        return max_profit
        

if __name__ == "__main__":
    s = Solution()
    print(s.maxProfit([7,1,5,3,6,4]))
    print(s.maxProfit([7,6,4,3,1]))
    print(s.maxProfit([2,4,1]))