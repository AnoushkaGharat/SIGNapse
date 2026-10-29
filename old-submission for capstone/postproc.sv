module postproc #(
  parameter int ACC_W = 32,
  parameter int SH_W  = 6
)(
  input logic signed [ACC_W-1:0] acc,
  input logic signed [ACC_W-1:0] bias,
  input logic [SH_W-1:0] shift,




  input logic bias_en,
  input logic relu_en,
  input logic passthru,




  output logic signed [ACC_W-1:0] res32,
  output logic signed [7:0] act8
);




  logic signed [ACC_W:0] biased;
  logic signed [ACC_W:0] half;
  logic signed [ACC_W:0] rnd;
  logic signed [ACC_W:0] shf;
  logic signed [ACC_W:0] relu;




  always_comb begin




     // Explicitly extend to ACC_W+1 bits.
     biased = $signed({acc[ACC_W-1], acc}) +
              (bias_en ? $signed({bias[ACC_W-1], bias}) : '0);




     half = (shift == '0) ? '0 :
            ($signed(1) <<< (shift - 1));




     rnd = biased + half;




     shf = rnd >>> shift;




     relu = (relu_en && (shf < 0)) ? '0 : shf;




     if (relu > 127)
         act8 = 8'sd127;
     else if (relu < -128)
         act8 = -8'sd128;
     else
         act8 = relu[7:0];




     res32 = passthru ? biased[ACC_W-1:0]
                      : {{(ACC_W-8){act8[7]}}, act8};




  end




endmodule











