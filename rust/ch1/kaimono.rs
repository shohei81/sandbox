fn main() {
    let pc_price = 98000.0;
    let a_ship_fee = 1200.0;
    let a_rate = 0.8;
    let b_ship_fee = 0.0;
    let b_rate = 0.9;
    println!("A社={}円", pc_price * a_rate + a_ship_fee);
    println!("B社={}円", pc_price * b_rate + b_ship_fee);
}
