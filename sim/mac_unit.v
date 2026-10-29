module mac_unit (
    input  wire               clk,
    input  wire               rst,
    input  wire               validity,
    input  wire               first_num,
    input  wire signed [7:0]  a,
    input  wire signed [7:0]  b,
    output reg  signed [31:0] acc
);
    wire signed [15:0] prod = a * b;

    always @(posedge clk) begin
        if (rst)
            acc <= 32'sd0;
        else if (validity)
            acc <= (first_num ? 32'sd0 : acc) + {{16{prod[15]}}, prod};
    end
endmodule
