module postproc (
    input  wire signed [31:0] acc,
    input  wire signed [31:0] bias,
    input  wire        [4:0]  shift,
    input  wire               relu,
    output reg  signed [7:0]  out
);
    reg signed [33:0] sum;
    reg signed [33:0] rnd;
    reg signed [33:0] r;

    always @* begin
        sum = {{2{acc[31]}}, acc} + {{2{bias[31]}}, bias};
        rnd = (shift == 5'd0) ? 34'sd0 : (34'sd1 <<< (shift - 5'd1));
        r   = (sum + rnd) >>> shift;
        if (relu && r < 0)
            r = 34'sd0;
        if (r > 34'sd127)
            out = 8'sd127;
        else if (r < -34'sd128)
            out = -8'sd128;
        else
            out = r[7:0];
    end
endmodule
