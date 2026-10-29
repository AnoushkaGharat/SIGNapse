module mac_unit #(parameter DW = 8, parameter ACC_W = 32) (
    input  logic clk,
    input  logic rst_n,
    input  logic signed [DW-1:0] a,
    input  logic signed [DW-1:0] b,
    input  logic valid_in,
    input  logic clear_acc,

    output logic signed [ACC_W-1:0] acc_out,
    output logic valid_out
);

    //1
    logic signed [2*DW-1:0] product_s1;
    logic valid_s1;
    logic first_term_s1;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            product_s1    <= 0;
            valid_s1      <= 0;
            first_term_s1 <= 0;
        end else begin
            product_s1    <= a * b;
            valid_s1      <= valid_in;
            first_term_s1 <= clear_acc;
        end
    end

   //2
    logic signed [ACC_W-1:0] acc;
    logic valid_s2;   

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc      <= 0;
            valid_s2 <= 0;             
        end else begin
            valid_s2 <= valid_s1;      
            if (valid_s1) begin
                if (first_term_s1)
                    acc <= {{(ACC_W-2*DW){product_s1[2*DW-1]}}, product_s1};
                else
                    acc <= acc + {{(ACC_W-2*DW){product_s1[2*DW-1]}}, product_s1};
                end
            end
    end

assign acc_out  = acc;
assign valid_out = valid_s2;   
endmodule